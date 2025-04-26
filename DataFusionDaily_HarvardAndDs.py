import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import os


def save_to_csv(df, output_path):
    """ 将 DataFrame 保存为 CSV 文件 """
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")


def extract_harvard_nc_to_dataframe(nc_file):
    """
    从哈佛 NetCDF 文件中提取臭氧数据，计算指标，并返回 DataFrame。

    参数:
    - nc_file: 哈佛 NetCDF 文件的路径。

    返回:
    - 包含臭氧指标的 DataFrame，用于合并。
    """
    # 手动指定年份（假设为单一年份数据集）
    year = '2011'

    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        rows = f.dimensions['ROW'].size  # 网格行数
        cols = f.dimensions['COL'].size  # 网格列数
        tstep = len(f.dimensions['TSTEP'])  # 时间步长（每日数据）

        # 提取臭氧变量
        ozone_data = f.variables['MDA8_O3'][:]

    # 创建 DataFrame 来存储提取的数据
    data = {
        'ROW': [],
        'COL': [],
        'Ozone': [],
        'Timestamp': []
    }

    for i in tqdm(range(tstep), desc="处理时间步长"):
        # 生成日期字符串
        date = pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')

        for row in range(rows):
            for col in range(cols):
                data['ROW'].append(row + 1)  # 索引从 1 开始
                data['COL'].append(col + 1)
                data['Ozone'].append(ozone_data[i, 0, row, col].item())
                data['Timestamp'].append(date_str)

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    # 计算关键臭氧相关指标
    return compute_ozone_metrics(df, 'harvard_ml')


def extract_ds_nc_to_dataframe(nc_file):
    """
    从 DS NetCDF 文件中提取臭氧数据，计算指标，并返回 DataFrame。

    参数:
    - nc_file: DS NetCDF 文件的路径。

    返回:
    - 包含臭氧指标的 DataFrame，用于合并。
    """
    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        rows = f.dimensions['ROW'].size  # 获取ROW维度的大小
        cols = f.dimensions['COL'].size  # 获取COL维度的大小
        tstep = len(f.dimensions['TSTEP'])  # 获取TSTEP维度的大小，即时间步长
        tflag = f.variables['TFLAG'][:]  # 获取TFLAG时间数据
        mda_o3 = f.variables['MDA_O3'][:]  # 获取MDA_O3变量数据

        # 从文件的全局属性中获取年份信息
        sdate = f.getncattr('SDATE')  # 获取SDATE属性
        year = str(sdate)[:4]  # 假设年份部分在SDATE的前四个字符

    # 将TFLAG时间转换为日期格式
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

    # 筛选一年的日期
    july_dates = [date for date in dates if date.startswith(f'{year}')]

    # 构建一个DataFrame并提取7月的数据
    data = {
        'ROW': [],
        'COL': [],
        'Ozone': [],
        'Timestamp': []
    }

    for i in tqdm(range(tstep)):
        if dates[i] not in july_dates:
            continue
        for row in range(rows):
            for col in range(cols):
                data['ROW'].append(row + 1)  # 行和列从1开始
                data['COL'].append(col + 1)
                data['Ozone'].append(mda_o3[i, 0, row, col].item())
                data['Timestamp'].append(dates[i])  # 对应的日期

    # 将数据保存到DataFrame中
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    # 计算关键臭氧相关指标
    return compute_ozone_metrics(df, 'ds_ozone')


def compute_ozone_metrics(df_data, variable_name):
    """
    计算臭氧相关指标，包括前 10 个 MDA8 平均值、季节性和年度平均值。

    参数:
    - df_data: 包含臭氧浓度数据的 DataFrame。
    - variable_name: 变量名，用于重命名列。

    返回:
    - 包含臭氧指标的 DataFrame，用于合并。
    """
    # 前 10 个 MDA8 臭氧日的平均值
    def top_10_avg(series):
        return series.nlargest(10).mean()

    df_top10 = df_data.groupby(["ROW", "COL"]).agg(
        {'Ozone': top_10_avg}
    ).reset_index()
    df_top10["Period"] = "top-10"

    # MDA8 的年度平均值
    df_annual = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'Ozone': 'mean'}
    ).reset_index()
    df_annual["Period"] = "Annual"

    # MDA8 的夏季（4 - 9 月）平均值
    summer_months = [4, 5, 6, 7, 8, 9]
    df_summer = df_data[df_data['Month'].isin(summer_months)]
    df_summer_avg = df_summer.groupby(["ROW", "COL"]).agg(
        {'Ozone': 'mean'}
    ).reset_index()
    df_summer_avg["Period"] = "Apr-Sep"

    # 季节性平均值（DJF, MAM, JJA, SON）
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    season_dfs = []
    for season, months in seasons.items():
        df_season = df_data[df_data['Month'].isin(months)]
        df_season_avg = df_season.groupby(["ROW", "COL"]).agg(
            {'Ozone': 'mean'}
        ).reset_index()
        df_season_avg["Period"] = season
        season_dfs.append(df_season_avg)

    # 将所有指标合并到一个 DataFrame 中
    final_df = pd.concat([df_top10, df_annual, df_summer_avg] + season_dfs, ignore_index=True)

    # 命名臭氧列以进行合并
    new_column_name = f'{variable_name}'
    final_df.rename(columns={'Ozone': new_column_name}, inplace=True)

    return final_df


def merge_with_existing(existing_df_path, new_df):
    """
    直接在原数据表上拼接新数据，并覆盖原文件。

    参数:
    - existing_df_path: 原始数据表（CSV 文件路径）
    - new_df: 计算出的新数据（DataFrame）

    作用:
    - 通过 ROW、COL、Period 匹配数据
    - 直接覆盖原文件，避免生成新的 CSV
    """
    # 读取现有数据表
    existing_df = pd.read_csv(existing_df_path)

    # 通过 ROW、COL、Period 进行匹配，拼接新数据
    existing_df = existing_df.merge(new_df, on=['ROW', 'COL', 'Period'], how='left')

    # 直接覆盖原数据表
    existing_df.to_csv(existing_df_path, index=False)
    print(f"数据已更新并覆盖原文件: {existing_df_path}")


# 定义路径
harvard_nc_file = '/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc'
ds_nc_file = '/backupdata/data_EPA/EQUATES/DS_data/CMAQv532_DSFusion_12US1_2011.nc'
existing_df_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/2011_Data_WithoutCV_Metrics.csv'

# 处理哈佛 ML 数据并合并
harvard_df = extract_harvard_nc_to_dataframe(harvard_nc_file)
merge_with_existing(existing_df_path, harvard_df)

# 处理 ds_ozone 数据并合并
ds_df = extract_ds_nc_to_dataframe(ds_nc_file)
merge_with_existing(existing_df_path, ds_df)
    