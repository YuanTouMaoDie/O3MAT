import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator
from scipy.spatial import KDTree
from functools import partial


def fill_missing_timezone_info(df_obs, df_gmt_offset):
    # 合并两个 DataFrame
    df_merged = pd.merge(df_obs, df_gmt_offset, on='site_id', how='left')
    # 找出缺失时区信息的行
    missing_rows = df_merged[df_merged['gmt_offset'].isna()]
    if len(missing_rows) == 0:
        print("所有的 site_id 在 df_gmt_offset 中都能找到对应信息，无需进行额外处理。")
        df_merged = df_merged.rename(columns={'Lat_x': 'Lat', 'Lon_x': 'Lon'})
        df_merged = df_merged.drop(columns=['Lat_y', 'Lon_y'])
        return df_merged

    print(f"存在 {len(missing_rows)} 个 site_id 在 df_gmt_offset 中找不到对应信息，将使用 KDTree 查找最近点的时区信息。")
    # 输出缺失行的 id 和经纬度
    missing_info = missing_rows[['site_id', 'Lat_x', 'Lon_x']]
    print("缺失时区信息的站点 id 和经纬度：")
    print(missing_info)

    # 构建 KDTree，使用合并后 DataFrame 中有时区信息的行的经纬度
    valid_rows = df_merged[~df_merged['gmt_offset'].isna()]
    tree = KDTree(valid_rows[['Lat_y', 'Lon_y']])
    for idx, row in missing_rows.iterrows():
        missing_lat = row['Lat_x']
        missing_lon = row['Lon_x']
        # 查询最近点的索引
        _, index = tree.query([missing_lat, missing_lon])
        # 确保 index 是整数
        if isinstance(index, np.ndarray):
            index = index[0]
        # 获取最近点的时区信息
        nearest_gmt_offset = valid_rows.iloc[index]['gmt_offset']
        nearest_epa_region = valid_rows.iloc[index]['epa_region']
        # 将最近点的时区信息填充到合并后的 DataFrame 中
        df_merged.loc[idx, 'gmt_offset'] = nearest_gmt_offset
        df_merged.loc[idx, 'epa_region'] = nearest_epa_region
    # 只保留一组经纬度信息
    df_merged = df_merged.rename(columns={'Lat_x': 'Lat', 'Lon_x': 'Lon'})
    df_merged = df_merged.drop(columns=['Lat_y', 'Lon_y'])

    # 筛选出之前缺失时区信息的行现在的时区信息
    filled_missing = df_merged.loc[missing_rows.index, ['gmt_offset', 'epa_region']]
    # 打印唯一值
    print("缺失点获取到的时区信息（唯一值）：")
    print(filled_missing.drop_duplicates())

    return df_merged


def process_date(date, df_obs_grouped, ds_models, model_pollutant):
    start_time = time.time()
    # 将numpy.datetime64转换为datetime对象
    date = pd.Timestamp(date).to_pydatetime()

    # Model的时间本身就是UTC
    adjusted_date = date
    print(f"UTC: {adjusted_date}\n")

    # 获取当天的监测数据
    df_daily_obs = df_obs_grouped[df_obs_grouped["utc_dateon"] == date].copy()
    year = adjusted_date.year
    month = adjusted_date.month

    # 根据 UTC 时间的年份和月份选择模型文件,适用全年
    file_index = (year - 2011) * 12 + month - 1
    if file_index < 0 or file_index >= len(ds_models):
        print(f"警告：没有对应的模型文件，UTC 日期: {adjusted_date}，跳过处理。")
        return pd.DataFrame()

    # 获取对应月份的模型数据
    ds_model = ds_models[file_index]
    proj = pyproj.Proj(ds_model.crs_proj4)

    # 将经纬度转换为模型的x, y坐标
    df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

    # 获取当天的模型数据
    tstep_value = pd.Timestamp(f"{adjusted_date.year}-{month:02d}-{adjusted_date.day:02d} {adjusted_date.hour:02d}:00:00")
    try:
        ds_hourly_model = ds_model.sel(TSTEP=tstep_value)
    except KeyError:
        print(f"警告：对应的 UTC 日期 {adjusted_date} 的模型数据缺失，跳过处理。")
        return pd.DataFrame()

    # 选择模型数据，并计算偏差
    df_daily_obs["mod"] = ds_hourly_model[model_pollutant][0].sel(
        ROW=df_daily_obs["y"].to_xarray(),
        COL=df_daily_obs["x"].to_xarray(),
        method="nearest"
    )
    df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["O3"]
    df_daily_obs["r_n"] = df_daily_obs["O3"] / df_daily_obs["mod"]

    # 筛选出 r_n > 5 的数据
    rn_filtered = df_daily_obs[df_daily_obs["r_n"] > 5].copy()
    rn_filtered['Timestamp'] = date.strftime('%Y-%m-%d %H:%M:%S')
    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing data for {date} took {duration:.2f} seconds")
    return rn_filtered[['site_id', 'Timestamp', 'Lat', 'Lon', 'O3', 'mod', 'r_n', 'gmt_offset']]


@timer_decorator
def start_hourly_data_fusion(model_files, monitor_file_template, region_table_file, time_region_file, file_path,
                             monitor_pollutant="O3", model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file_template: 监测文件模板，使用 {year} 占位符
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} time_region_file: 包含站点ID和TimeRegion两列的文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    """
    if start_date and end_date:
        start = pd.to_datetime(start_date.replace('/', '-'))
        end = pd.to_datetime(end_date.replace('/', '-'))
        filter_start = start - pd.Timedelta(hours=8)
        filter_end = end - pd.Timedelta(hours=4)

        # 如果是UTC第一天，左偏移读去年监测数据，同时考虑end year第一天数据
        years_to_read = [start.year]
        if start.month == 1 and start.day == 1:
            years_to_read.append(start.year - 1)
        if end.month == 1 and end.day == 1:
            years_to_read.append(end.year)

        df_obs_list = []
        for year in years_to_read:
            monitor_file = monitor_file_template.format(year=year)
            df = pd.read_csv(monitor_file)
            df['dateon'] = pd.to_datetime(df['dateon'])
            df = df[(df['dateon'] >= filter_start) & (df['dateon'] <= filter_end)]
            df_obs_list.append(df)
        df_obs = pd.concat(df_obs_list, ignore_index=True)

    else:
        # 如果没有指定日期范围，按原逻辑读取数据
        monitor_file = monitor_file_template.format(year=2011)
        df_obs = pd.read_csv(monitor_file)
        df_obs['dateon'] = pd.to_datetime(df_obs['dateon'])

    # 读取时区偏移量信息,标准时间-5~-8
    df_gmt_offset = pd.read_csv('output/Region/MonitorsTimeRegion_Filter_ST.csv')

    # 调用函数填充缺失的时区信息，一般不会缺失
    df_obs = fill_missing_timezone_info(df_obs, df_gmt_offset)

    # 监测点的ST转为对应UTC，比如最东部UTC=ST - (-5)
    df_obs['utc_dateon'] = df_obs['dateon'] - pd.to_timedelta(df_obs['gmt_offset'], unit='h')

    # 处理日期范围
    if start_date and end_date:
        start = pd.to_datetime(start_date.replace('/', '-'))  # 修改日期格式
        end = pd.to_datetime(end_date.replace('/', '-'))  # 修改日期格式
        df_obs = df_obs[(df_obs['utc_dateon'] >= start) & (df_obs['utc_dateon'] <= end)]

    # 按站点和日期聚合
    df_obs_grouped = (
        df_obs.groupby(["site_id", "utc_dateon"])
        .agg({"O3": "mean", "Lat": "mean", "Lon": "mean", "gmt_offset": "first"})
        .reset_index()
    )
    dates = df_obs_grouped["utc_dateon"].unique()

    # sort dates,如果不sort的话当日期范围跨过3月时就会出现从3月开始的现象
    dates = np.sort(dates)

    # 一次性读取所有模型文件
    ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]

    total_hours = len(dates)
    print(f"总共有 {total_hours} 个小时的数据需要处理。")

    all_rn_filtered = []
    for i, date in enumerate(dates):
        result = process_date(date, df_obs_grouped, ds_models, model_pollutant)
        if not result.empty:
            all_rn_filtered.append(result)
        print(f"已经处理了 {i + 1} 个小时的数据，还剩 {total_hours - (i + 1)} 个小时的数据待处理。")

    # 合并所有筛选后的数据
    df_rn_filtered = pd.concat(all_rn_filtered, ignore_index=True)

    # 将 Timestamp 从 UTC 时间转换为本地时间
    df_rn_filtered['Timestamp'] = pd.to_datetime(df_rn_filtered['Timestamp'])
    df_rn_filtered['localtime'] = df_rn_filtered['Timestamp'] + pd.to_timedelta(df_rn_filtered['gmt_offset'], unit='h')

    # 筛选出 2011 年的数据
    df_rn_filtered_2011 = df_rn_filtered[df_rn_filtered['localtime'].dt.year == 2011]

    # 筛选出 2011 年 3 月到 10 月的数据
    df_rn_filtered_2011 = df_rn_filtered_2011[
        (df_rn_filtered_2011['localtime'].dt.month >= 3) & (df_rn_filtered_2011['localtime'].dt.month <= 10)]

    # 保存筛选后的数据表
    rn_filtered_file_path = os.path.join(os.path.dirname(file_path), 'rn_filtered_data.csv')
    df_rn_filtered_2011.to_csv(rn_filtered_file_path, index=False)
    print(f"Filtered data with r_n > 5 for 2011 (March to October) saved to {rn_filtered_file_path}")

    # 计算每个站点的平均值
    df_mean_by_site = df_rn_filtered_2011.groupby('site_id').mean()
    df_mean_by_site = df_mean_by_site.reset_index()

    # 保存平均值数据表
    mean_file_path = os.path.join(os.path.dirname(file_path), 'mean_data_by_site.csv')
    df_mean_by_site.to_csv(mean_file_path, index=False)
    print(f"Average data by site for 2011 (March to October) saved to {mean_file_path}")

    return df_rn_filtered_2011, df_mean_by_site


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_files = [
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201101.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201102.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201103.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201104.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201105.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201106.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201107.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201108.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201109.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201110.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201111.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201112.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201201.nc"
    ]

    # 使用占位符 {year} 表示年份
    monitor_file_template = r"/backupdata/data_EPA/aq_obs/routine/{year}/AQS_hourly_data_{year}_LatLon.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    time_region_file = r"output/Region/MonitorsTimeRegion_Filter_ST.csv"  # 请替换为实际的TimeRegion文件路径

    # # # 指定日期范围
    start_date = '2011/03/01 00:00'
    end_date = '2011/11/01 08:00'

    # start_date = '2011/03/01 00:00'
    # end_date = '2011/03/01 08:00'

    daily_output_path = os.path.join(save_path, "2011_HourlyData_r_n.csv")
    filtered_data, mean_data = start_hourly_data_fusion(
        model_files,
        monitor_file_template,
        region_table_file,
        time_region_file,
        daily_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date,
    )
    print("Done!")