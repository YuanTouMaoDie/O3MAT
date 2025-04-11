import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator
import multiprocessing  # 用于获取CPU核心数


@timer_decorator
def start_hourly_data_fusion(model_files, monitor_file_2010, monitor_file_2011, region_table_file, time_region_file, file_path,
                             monitor_pollutant="O3", model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file_2010: 2010年监测文件，必须有列：site_id, POCode, dateon, dateoff, O3
    @param {string} monitor_file_2011: 2011年监测文件，必须有列：site_id, POCode, dateon, dateoff, O3
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} time_region_file: 包含站点ID和TimeRegion两列的文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    """
    # 读取2010年和2011年监测数据并合并
    df_obs_2010 = pd.read_csv(monitor_file_2010)
    df_obs_2011 = pd.read_csv(monitor_file_2011)
    df_obs = pd.concat([df_obs_2010, df_obs_2011], ignore_index=True)

    df_obs['dateon'] = pd.to_datetime(df_obs['dateon'])

    # 读取时区信息
    time_region_df = pd.read_csv(time_region_file)
    timezone_mapping = {'PST': 0, 'MST': 1, 'CST': 2, 'EST': 3}
    df_obs = df_obs.merge(time_region_df, on='site_id', how='left')
    df_obs['offset'] = df_obs['TimeRegion'].map(timezone_mapping)
    df_obs['dateon_adjusted'] = df_obs.apply(
        lambda row: row['dateon'] - pd.Timedelta(hours=row['offset']), axis=1
    )

    # 处理日期范围
    if start_date and end_date:
        start = pd.to_datetime(start_date.replace('/', '-'))  # 修改日期格式
        end = pd.to_datetime(end_date.replace('/', '-'))  # 修改日期格式
        df_obs = df_obs[(df_obs['dateon_adjusted'] >= start) & (df_obs['dateon_adjusted'] <= end)]

    # 按站点和日期聚合
    df_obs_grouped = (
        df_obs.groupby(["site_id", "dateon_adjusted"])
       .agg({"O3": "mean", "Lat": "mean", "Lon": "mean"})
       .reset_index()
    )
    dates = df_obs_grouped["dateon_adjusted"].unique()

    # sort dates,如果不sort的话当日期范围跨过3月时就会出现从3月开始的现象
    dates = np.sort(dates)

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)

    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1]
    # 进行偏移以确保正确的坐标
    us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 获取模型的行列坐标
    us_region_row_col = us_region_df[['COL', 'ROW']].values

    # 一次性读取所有模型文件
    ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]

    # 使用NNA进行站点数据的匹配
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    all_results = []

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            # 将numpy.datetime64转换为datetime对象
            date = pd.Timestamp(date).to_pydatetime()

            # 模型时间：PST时间加8小时得到对应的UTC时间
            adjusted_date = date + pd.Timedelta(hours=8)
            print(f"监测日期（以PST为基准调整后）: {date}，对应的模型日期（UTC 时间）: {adjusted_date}")

            # 获取当天的监测数据
            df_daily_obs = df_obs_grouped[df_obs_grouped["dateon_adjusted"] == date].copy()
            year = adjusted_date.year
            month = adjusted_date.month

            # 根据 UTC 时间的年份和月份选择模型文件
            file_index = (year - 2011) * 12 + month - 1
            if file_index < 0 or file_index >= len(ds_models):
                print(f"警告：没有对应的模型文件，UTC 日期: {adjusted_date}，跳过处理。")
                continue

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
                continue

            # 选择模型数据，并计算偏差
            df_daily_obs["mod"] = ds_hourly_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest"
            )
            df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["O3"]
            df_daily_obs["r_n"] = df_daily_obs["O3"] / df_daily_obs["mod"]

            df_prediction = ds_hourly_model[["ROW", "COL"]].to_dataframe().reset_index()

            # 训练NNA模型
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            # 并行计算
            njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
            zdf = nn.predict(us_region_row_col, njobs=njobs)

            # 创建一个全为NaN的DataFrame来存储预测结果
            result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"])

            # 将美国区域的预测结果填充到对应的行
            result_df.loc[us_region_df.index] = zdf

            df_prediction = pd.concat([df_prediction, result_df], axis=1)

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_hourly_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_hourly_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_hourly_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_hourly_model[model_pollutant][0].values.flatten()
            # 修改日期格式为带小时的形式
            df_fusion["TSTEP"] = adjusted_date.strftime('%Y-%m-%d %H:%M:%S')  # 使用调整后的 UTC 时间
            df_fusion["Timestamp"] = date.strftime('%Y-%m-%d %H:%M:%S')
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            all_results.append(df_fusion)

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    # 一次性合并所有结果
    df_all_hourly_prediction = pd.concat(all_results, ignore_index=True)

    df_all_hourly_prediction.to_csv(file_path, index=False)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")
    return df_all_hourly_prediction


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

    monitor_file_2010 = r"/backupdata/data_EPA/aq_obs/routine/2010/AQS_hourly_data_2010_LatLon.csv"
    monitor_file_2011 = r"/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011_LatLon.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    time_region_file = r"/path/to/TimeRegion.csv"  # 请替换为实际的TimeRegion表路径

    # 指定日期范围，提前到2010年12月31日18:00
    start_date = '2010/12/31 18:00'
    end_date = '2011/12/31 23:00'

    daily_output_path = os.path.join(save_path, "2011_SixDataset_Hourly_True_5Hours.csv")
    start_hourly_data_fusion(
        model_files,
        monitor_file_2010,
        monitor_file_2011,
        region_table_file,
        time_region_file,
        daily_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date,
    )
    print("Done!")