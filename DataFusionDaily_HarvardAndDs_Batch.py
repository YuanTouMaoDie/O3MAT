import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor


def save_to_csv(df, output_path):
    """ 将 DataFrame 保存为 CSV 文件 """
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")


def extract_harvard_nc_to_dataframe(nc_file, year):
    """
    从哈佛 NetCDF 文件中提取臭氧数据，计算指标，并返回 DataFrame。

    参数:
    - nc_file: 哈佛 NetCDF 文件的路径。
    - year: 指定的年份

    返回:
    - 包含臭氧指标的 DataFrame，用于合并。
    """
    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        rows = f.dimensions['ROW'].size  # 网格行数
        cols = f.dimensions['COL'].size  # 网格列数
        tstep = len(f.dimensions['TSTEP'])  # 时间步长（每日数据）

        # 提取臭氧变量
        ozone_data = f.variables['MDA8_O3'][:]

    # 生成日期数组
    dates = pd.date_range(start=f'{year}-01-01', periods=tstep)

    rows_arr = []
    cols_arr = []
    ozone_arr = []
    dates_arr = []

    # 使用 tqdm 显示每天的处理进度
    for day in tqdm(range(tstep), desc=f"处理哈佛数据 {year} 年天数"):
        rows_arr.extend(np.tile(np.arange(1, rows + 1), cols))
        cols_arr.extend(np.repeat(np.arange(1, cols + 1), rows))
        ozone_arr.extend(ozone_data[day].flatten())
        dates_arr.extend([dates[day]] * cols * rows)

    data = {
        'ROW': rows_arr,
        'COL': cols_arr,
        'Ozone': ozone_arr,
        'Timestamp': dates_arr
    }

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    # 计算关键臭氧相关指标
    return compute_ozone_metrics_parallel(df, 'harvard_ml')


def extract_ds_nc_to_dataframe(nc_file, year):
    """
    从 DS NetCDF 文件中提取臭氧数据，计算指标，并返回 DataFrame。

    参数:
    - nc_file: DS NetCDF 文件的路径。
    - year: 指定的年份

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

    # 将TFLAG时间转换为日期格式
    tflag_year_day = tflag[:, 0, 0].astype(int)
    day_of_year = tflag_year_day % 1000
    valid_days_mask = day_of_year <= 366
    dates = np.empty(tstep, dtype=object)
    valid_dates = pd.to_datetime([f'{year}{day:03}' for day in day_of_year[valid_days_mask]], format='%Y%j')
    dates[valid_days_mask] = valid_dates.strftime('%Y-%m-%d')
    dates[~valid_days_mask] = "Invalid day_of_year"

    # 筛选一年的日期
    july_dates = [date for date in dates if date.startswith(f'{year}')]

    rows_arr = []
    cols_arr = []
    ozone_arr = []
    dates_arr = []

    # 使用 tqdm 显示每天的处理进度
    for day in tqdm(range(tstep), desc=f"处理 DS 数据 {year} 年天数"):
        rows_arr.extend(np.tile(np.arange(1, rows + 1), cols))
        cols_arr.extend(np.repeat(np.arange(1, cols + 1), rows))
        ozone_arr.extend(mda_o3[day].flatten())
        dates_arr.extend([dates[day]] * rows * cols)

    data = {
        'ROW': rows_arr,
        'COL': cols_arr,
        'Ozone': ozone_arr,
        'Timestamp': dates_arr
    }

    df = pd.DataFrame(data)
    df = df[df['Timestamp'].isin(july_dates)]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month

    # 计算关键臭氧相关指标
    return compute_ozone_metrics_parallel(df, 'ds_ozone')


def compute_ozone_metrics_single_grid(grid_df, variable_name):
    """
    计算单个网格的臭氧相关指标
    """
    # 前 10 个 MDA8 臭氧日的平均值
    def top_10_avg(series):
        return series.nlargest(10).mean()

    df_top10 = pd.DataFrame({
        'ROW': [grid_df['ROW'].iloc[0]],
        'COL': [grid_df['COL'].iloc[0]],
        f'{variable_name}': [top_10_avg(grid_df['Ozone'])],
        'Period': ['top-10']
    })

    # MDA8 的年度平均值
    df_annual = pd.DataFrame({
        'ROW': [grid_df['ROW'].iloc[0]],
        'COL': [grid_df['COL'].iloc[0]],
        'Year': [grid_df['Year'].iloc[0]],
        f'{variable_name}': [grid_df['Ozone'].mean()],
        'Period': ['Annual']
    })

    # MDA8 的夏季（4 - 9 月）平均值
    summer_months = [4, 5, 6, 7, 8, 9]
    df_summer = grid_df[grid_df['Month'].isin(summer_months)]
    df_summer_avg = pd.DataFrame({
        'ROW': [grid_df['ROW'].iloc[0]],
        'COL': [grid_df['COL'].iloc[0]],
        f'{variable_name}': [df_summer['Ozone'].mean()],
        'Period': ['Apr-Sep']
    })

    # 季节性平均值（DJF, MAM, JJA, SON）
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    season_dfs = []
    for season, months in seasons.items():
        df_season = grid_df[grid_df['Month'].isin(months)]
        df_season_avg = pd.DataFrame({
            'ROW': [grid_df['ROW'].iloc[0]],
            'COL': [grid_df['COL'].iloc[0]],
            f'{variable_name}': [df_season['Ozone'].mean()],
            'Period': [season]
        })
        season_dfs.append(df_season_avg)

    # 将所有指标合并到一个 DataFrame 中
    final_df = pd.concat([df_top10, df_annual, df_summer_avg] + season_dfs, ignore_index=True)

    return final_df


def compute_ozone_metrics_parallel(df_data, variable_name):
    """
    并行计算每个网格的臭氧相关指标
    """
    unique_grids = df_data.groupby(['ROW', 'COL'])
    num_grids = len(unique_grids)
    print(f"开始处理 {num_grids} 个网格")

    with ProcessPoolExecutor() as executor:
        futures = []
        for _, grid_df in unique_grids:
            future = executor.submit(compute_ozone_metrics_single_grid, grid_df, variable_name)
            futures.append(future)

        results = [future.result() for future in futures]

    final_df = pd.concat(results, ignore_index=True)
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


def process_nc_file(nc_file, extract_func, year):
    return extract_func(nc_file, year)


# 指定年份范围
start_year = 2011
end_year = 2011  # 可根据需要修改结束年份

for year in range(start_year, end_year + 1):
    print(f"开始处理 {year} 年的数据")
    # 动态生成文件路径
    harvard_nc_file = f'/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_{year}_12km.nc'
    ds_nc_file = f'/backupdata/data_EPA/EQUATES/DS_data/CMAQv532_DSFusion_12US1_{year}.nc'
    existing_df_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/{year}_Data_WithoutCV_Metrics.csv'

    # 并行处理哈佛 ML 数据和 ds_ozone 数据
    with ProcessPoolExecutor() as executor:
        future_harvard = executor.submit(process_nc_file, harvard_nc_file, extract_harvard_nc_to_dataframe, year)
        future_ds = executor.submit(process_nc_file, ds_nc_file, extract_ds_nc_to_dataframe, year)

        harvard_df = future_harvard.result()
        ds_df = future_ds.result()

    merge_with_existing(existing_df_path, harvard_df)
    merge_with_existing(existing_df_path, ds_df)
    