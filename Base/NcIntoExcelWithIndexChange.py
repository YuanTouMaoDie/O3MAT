import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import os


def save_to_csv(df, output_path):
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")


def extract_nc_to_dataframe(nc_file, enable_metrics=False, save_path=None, project_name=None):
    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        # 获取维度信息
        rows = f.dimensions['ROW'].size  # 获取ROW维度的大小
        cols = f.dimensions['COL'].size  # 获取COL维度的大小
        tstep = len(f.dimensions['TSTEP'])  # 获取TSTEP维度的大小，即时间步长
        var = f.dimensions['VAR']  # 获取变量维度
        date_time = f.dimensions['DATE-TIME']  # 获取时间

        # 获取变量数据
        mda_o3 = f.variables['MDA_O3'][:]  # 获取MDA_O3变量数据，四维数组 (TSTEP, LAY, ROW, COL),Model中的是O3_MDA8
        # mda_o31 = f.variables['eVNA_Y'][:]  # 获取MDA_O3变量数据，四维数组 (TSTEP, LAY, ROW, COL),Model中的是O3_MDA8
        tflag = f.variables['TFLAG'][:]  # 获取TFLAG时间数据

        # 从文件的全局属性中获取年份信息
        sdate = f.getncattr('SDATE')  # 获取SDATE属性
        year = str(sdate)[:4]  # 假设年份部分在SDATE的前四个字符

        # 将TFLAG时间转换为日期格式并打印
        dates = []
        for time in tflag[:, 0, 0]:  # TFLAG的第一个元素表示年份信息
            year_day = int(time)  # 提取年份和天数部分
            day_of_year = year_day % 1000  # 提取天数
            if day_of_year <= 366:  # 检查是否是有效的天数
                try:
                    # 使用年份和天数生成日期
                    date = pd.to_datetime(f'{year}{day_of_year:03}', format='%Y%j').strftime('%Y-%m-%d')
                    dates.append(date)
                except ValueError:
                    dates.append("Invalid day_of_year")  # 如果转换失败，则做标记
            else:
                dates.append("Invalid day_of_year")  # 对于无效的天数，做标记

        # 筛选7月的日期
        july_dates = [date for date in dates if date.startswith(f'{year}')]

        # 构建一个DataFrame并提取7月的数据
        data = {
            'ROW': [],
            'COL': [],
            'ds_ozone': [],
            'Timestamp': []
        }

        for i in tqdm(range(tstep)):
            if dates[i] not in july_dates:
                continue 
            for row in range(rows):
                for col in range(cols):
                    data['ROW'].append(row + 1)  # 行和列从1开始
                    data['COL'].append(col + 1)
                    data['ds_ozone'].append(mda_o3[i, 0, row, col].item())  # 使用 .item() 获取标量值
                    # data['evna_ozone'].append(mda_o31[i, 0, row, col].item())  # 使用 .item() 获取标量值
                    data['Timestamp'].append(dates[i])  # 对应的日期

        # 将数据保存到DataFrame中
        df = pd.DataFrame(data)

        if enable_metrics:
            if save_path is None or project_name is None:
                raise ValueError("When enabling metrics, save_path and project_name must be provided.")
            output_file_list = save_daily_data_fusion_to_metrics(df, save_path, project_name)
            print("Metrics files saved:", output_file_list)
        else:
            # 输出CSV路径
            output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronResult_aVNA_2011_daily.csv'  # 请替换为实际路径
            save_to_csv(df, output_path)


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

    # 初始化一个空的 DataFrame 来存储所有指标
    all_metrics = []

    # top-10 average of MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()

    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'ds_ozone': top_10_average}
    ).reset_index()
    df_data_top_10_avg["Period"] = f"top-10"
    all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'ds_ozone': 'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'ds_ozone': 'mean'}
    ).reset_index()
    df_data_summer_avg["Period"] = f"Apr-Sep"
    all_metrics.append(df_data_summer_avg)

    # seasonal averages（DJF, MAM, JJA, SON）of MDA8
    seasons = {
        'DJF': [12, 1, 2],  # December,January, Feburary
        'MAM': [3, 4, 5],  # April, May, June
        'JJA': [6, 7, 8],  # July, August, September
        'SON': [9, 10, 11]  # October, November, December
    }
    for season, months in seasons.items():
        df_data_season = df_data[df_data['Month'].isin(months)]
        df_data_season_avg = df_data_season.groupby(["ROW", "COL"]).agg(
            {'ds_ozone': 'mean'}
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}_Index.csv")
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)

    return output_file_list


# 输入.nc文件路径
# nc_file = '/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc'  # 请替换为实际路径
nc_file = '/backupdata/data_EPA/EQUATES/DS_data/CMAQv532_DSFusion_12US1_2011.nc'  # 请替换为实际路径

# 提取数据并保存
enable_metrics = True  # 设置为 True 开启指标计算功能
save_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output'  # 请替换为实际路径
project_name = 'BarronResult_aVNA_FtAIndex_2011'  # 输出数据表的名称
extract_nc_to_dataframe(nc_file, enable_metrics=enable_metrics, save_path=save_path, project_name=project_name)