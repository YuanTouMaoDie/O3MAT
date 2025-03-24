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
def start_hourly_data_fusion(model_files, monitor_file, region_table_file, file_path, monitor_pollutant="Ozone",
                            model_pollutant="O3_MDA8", start_date=None, end_date=None, lat_lon_file=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file: 监测文件，必须有列：site_id, POCode, dateon, dateoff, O3
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} lat_lon_file: 包含 site_id, Lat, Lon 信息的文件
    """
    # 读取监测数据
    df_obs = pd.read_csv(monitor_file)
    df_obs['dateon'] = pd.to_datetime(df_obs['dateon'])
        
    # 处理日期范围
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df_obs = df_obs[(df_obs['dateon'] >= start) & (df_obs['dateon'] <= end)]
    
    # 按站点和日期聚合
    df_obs_grouped = (
        df_obs.groupby(["site_id", "dateon"])
        .agg({"O3": "mean", "Lat": "mean", "Lon": "mean"})
        .reset_index()
    )
    dates = df_obs_grouped["dateon"].unique()

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)

    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1]
    # 进行偏移以确保正确的坐标
    us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 获取模型的行列坐标
    us_region_row_col = us_region_df[['COL', 'ROW']].values
    
    # 使用NNA进行站点数据的匹配
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    df_all_hourly_prediction = None

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            # 获取当天的监测数据
            df_daily_obs = df_obs_grouped[df_obs_grouped["dateon"] == date].copy()
            month = date.month
            hour = date.hour

            # 遍历12个月的模型文件
            model_file = model_files[month - 1]  # 获取对应月份的模型文件
            ds_model = pyrsig.open_ioapi(model_file)
            proj = pyproj.Proj(ds_model.crs_proj4)

            # 将经纬度转换为模型的x, y坐标
            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

            # 获取当天的模型数据
            tstep_value = int(f"{date.year}{month:02d}{date.day:02d}{hour:02d}")
            ds_hourly_model = ds_model.sel(TSTEP=tstep_value)

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
            df_fusion["Timestamp"] = date
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            if df_all_hourly_prediction is None:
                df_all_hourly_prediction = df_fusion
            else:
                df_all_hourly_prediction = pd.concat([df_all_hourly_prediction, df_fusion])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    df_all_hourly_prediction.to_csv(file_path, index=False)
    project_name = os.path.basename(file_path).replace(".csv", "")
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

    # 调用save_daily_data_fusion_to_metrics函数
    save_path = os.path.dirname(file_path)
    output_file_list = save_daily_data_fusion_to_metrics(df_all_hourly_prediction, save_path, project_name)
    print(f"O3-related metrics files saved: {output_file_list}")
    return df_all_hourly_prediction


def save_daily_data_fusion_to_metrics(df_data, save_path, project_name):
    '''
    This function converts the daily data fusion results to O3-related metrics (e.g., 98th percentile of MDA8 ozone concentration, average of top-10 MDA8 ozone days, annual average of MDA8 ozone concentration) files.
    @param {DataFrame} df_data: The DataFrame of the daily data fusion results.
    @param {str} save_path: The path to save the O3-related metrics files.
    @param {str} project_name: The name of the project.
    @return {list} output_file_list: The list of the O3-related metrics files.
    '''
    output_file_list = []

    # 提取年份和月份
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month
    df_data['hour'] = df_data['Timestamp'].dt.hour

    # 筛选白天（8:00 AM 到 8:00 PM）的数据
    df_daytime_data = df_data[(df_data['hour'] >= 8) & (df_data['hour'] < 20)]

    # 计算加权 O3 浓度
    df_daytime_data['weighted_O3'] = df_daytime_data['vna_ozone'] / (1 + 4403 * np.exp(-126 * df_daytime_data['vna_ozone']))

    # 按月分组计算每月的 W126 指标
    df_monthly_w126 = df_daytime_data.groupby([df_daytime_data['Year'], df_daytime_data['Month']]).agg(
        {'weighted_O3':'sum', 'vna_ozone': 'count'}
    ).reset_index()
    df_monthly_w126.columns = ['year','month','monthly_w126', 'daytime_hours_count']

    # 处理缺失数据
    total_possible_daytime_hours = 31 * 12  # 假设一个月最多 31 天，每天 12 个白天小时
    df_monthly_w126['available_ratio'] = df_monthly_w126['daytime_hours_count'] / total_possible_daytime_hours
    df_monthly_w126 = df_monthly_w126[df_monthly_w126['available_ratio'] >= 0.75]
    df_monthly_w126['adjusted_monthly_w126'] = df_monthly_w126.apply(
        lambda row: row['monthly_w126'] * (1 / row['available_ratio']) if row['available_ratio'] < 1 else row['monthly_w126'],
        axis=1
    )

    # 计算移动 3 个月的总和
    three_month_sums = []
    for start_month in range(3, 9):
        end_month = start_month + 2
        subset = df_monthly_w126[(df_monthly_w126['month'] >= start_month) & (df_monthly_w126['month'] <= end_month)]
        if len(subset) == 3:
            three_month_sum = subset['adjusted_monthly_w126'].sum()
            three_month_sums.append({
               'start_month': start_month,
                'end_month': end_month,
                'three_month_sum': three_month_sum
            })
    df_three_month_sums = pd.DataFrame(three_month_sums)

    # 确定年度 W126 指标
    annual_w126 = df_three_month_sums['three_month_sum'].max()

    # 初始化一个空的 DataFrame 来存储所有指标
    all_metrics = []

    # top-10 average of MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()

    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': top_10_average,
         'evna_ozone': top_10_average,
         'avna_ozone': top_10_average,
        'model': top_10_average}
    ).reset_index()
    df_data_top_10_avg["Period"] = f"top-10"
    all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'vna_ozone':'mean',
         'evna_ozone':'mean',
         'avna_ozone':'mean',
        'model':'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'vna_ozone':'mean',
         'evna_ozone':'mean',
         'avna_ozone':'mean',
        'model':'mean'}
    ).reset_index()
    df_data_summer_avg["Period"] = f"Apr-Sep"
    all_metrics.append(df_data_summer_avg)

    # seasonal averages（DJF, MAM, JJA, SON）of MDA8
    seasons = {
        'DJF': [12, 1, 2],  # December, January, February
        'MAM': [3, 4, 5],  # March, April, May
        'JJA': [6, 7, 8],  # June, July, August
        'SON': [9, 10, 11]  # September, October, November
    }
    for season, months in seasons.items():
        df_data_season = df_data[df_data['Month'].isin(months)]
        df_data_season_avg = df_data_season.groupby(["ROW", "COL"]).agg(
            {'vna_ozone':'mean',
             'evna_ozone':'mean',
             'avna_ozone':'mean',
            'model':'mean'}
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 添加年度 W126 指标到指标列表
    w126_metric = pd.DataFrame({
        'Period': ['W126'],
        'Value': [annual_w126]
    })
    all_metrics.append(w126_metric)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}_metrics.csv")
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)

    return output_file_list


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_files = [
        r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/EQUATES_COMBINE_ACONC_O3_201101.nc",
        r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/EQUATES_COMBINE_ACONC_O3_201102.nc",
        r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/EQUATES_COMBINE_ACONC_O3_201103.nc",
        # 继续添加其他月份的文件路径...
    ]
    
    monitor_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/AQS_hourly_data_2011.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    lat_lon_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # 指定日期范围
    start_date = '2011/09/01 00:00'
    end_date = '2011/09/01 02:00'

    daily_output_path = os.path.join(save_path, "2011_SixDataset_Hourly.csv")
    start_hourly_data_fusion(
        model_files,
        monitor_file,
        region_table_file,
        daily_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date,
        lat_lon_file=lat_lon_file
    )
    print("Done!")
