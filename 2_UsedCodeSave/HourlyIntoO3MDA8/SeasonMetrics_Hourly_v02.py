import pandas as pd
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd
from dask import delayed, compute
import numba
from joblib import Parallel, delayed
import warnings


# 定义需要读取的列
usecols = ['ROW', 'COL', 'Timestamp', 'vna_ozone', 'avna_ozone', 'evna_ozone','model']

# 定义每列的数据类型，减少内存使用
dtype = {
    'ROW': 'int32',
    'COL': 'int32',
    'Timestamp': 'object',
    'vna_ozone': 'float32',
    'avna_ozone': 'float32',
    'evna_ozone': 'float32',
   'model': 'float32'
}


@numba.jit(nopython=True)
def rolling_mean_numba(arr, window):
    n = arr.shape[0]
    result = np.empty(n)
    for i in range(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(arr[i - window + 1:i + 1])
    return result


@delayed
def calculate_mda8(df):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("开始计算 MDA8 指标...")
    df['Date'] = df['Timestamp'].dt.date
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone','model']
    mda8_data = []
    grouped = df.groupby(['ROW', 'COL'])
    total_grids = len(grouped)
    with tqdm(total=total_grids, desc="Processing MDA8 ROW - COL grids") as pbar:
        for (row, col), group in grouped:
            daily_mda8 = {'ROW': row, 'COL': col}
            sub_grouped = group.groupby('Date')
            for date, sub_group in sub_grouped:
                for col_name in ozone_columns:
                    values = sub_group[col_name].values
                    rolling_means = rolling_mean_numba(values, 8)
                    if date not in daily_mda8:
                        daily_mda8[date] = {}
                    daily_mda8[date][col_name] = np.nanmax(rolling_means)
            mda8_data.append(daily_mda8)
            pbar.update(1)
    mda8_flat_data = []
    for item in mda8_data:
        row = item['ROW']
        col = item['COL']
        for date, ozone_values in item.items():
            if date in ['ROW', 'COL']:
                continue
            flat_item = {'ROW': row, 'COL': col, 'Date': date}
            flat_item.update(ozone_values)
            mda8_flat_data.append(flat_item)
    mda8_df = pd.DataFrame(mda8_flat_data)
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone','model']
    mda8_df['Year'] = mda8_df['Date'].apply(lambda x: x.year)
    mda8_df['Month'] = mda8_df['Date'].apply(lambda x: x.month)
    print("MDA8 指标计算完成.")
    return mda8_df


@delayed
def calculate_other_metrics(mda8_df, save_path, project_name):
    print("开始计算其他指标（如DJF等）...")

    def calculate_top_10_average(series):
        return series.nlargest(10).mean()

    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone','model']
    all_metrics = []
    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    periods = {
        'top-10': (calculate_top_10_average, ['ROW', 'COL'], mda8_df),
        'Annual': ('mean', ['ROW', 'COL', 'Year'], mda8_df),
        'Apr-Sep': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin([4, 5, 6, 7, 8, 9])]),
        'DJF': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['DJF'])]),
        'MAM': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['MAM'])]),
        'JJA': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['JJA'])]),
        'SON': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['SON'])])
    }

    def process_period_col(period, agg_func, groupby_cols, df, col):
        if isinstance(agg_func, str):
            result = df.groupby(groupby_cols)[col].agg(agg_func)
        else:
            result = df.groupby(groupby_cols)[col].apply(agg_func)
        result_df = result.reset_index()
        result_df['Period'] = period
        return result_df

    # 生成所有需要处理的任务列表
    tasks = [(period, agg_func, groupby_cols, df, col)
             for period, (agg_func, groupby_cols, df) in periods.items()
             for col in ozone_columns]

    # 并行处理每个周期和臭氧列的组合，并显示进度条
    all_metrics = Parallel(n_jobs=4)(
        delayed(process_period_col)(*task)
        for task in tqdm(tasks, desc="处理网格进度")
    )

    final_metrics = pd.concat(all_metrics, axis=0)

    columns_order = ['ROW', 'COL', 'Period'] + ozone_columns
    result_df = final_metrics[columns_order]

    output_file = f'{save_path}/{project_name}_metrics.csv'
    result_df.to_csv(output_file, index=False)
    print(f"其他指标（如DJF等）数据已保存到: {output_file}")
    print("其他指标（如DJF等）计算完成.")
    return output_file


def convert_all_to_local_time(df_data, timezone_df):
    """
    将所有数据的时间转换为本地时间，并添加年、月、日、小时列
    """
    # 合并数据
    merged_df = pd.merge(df_data, timezone_df, on=['ROW', 'COL'], how='left')
    # 过滤掉 gmt_offset 为 NaN 的行
    merged_df = merged_df.dropna(subset=['gmt_offset'])
    # 转换时间戳
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])

    # 使用向量化操作计算本地时间
    valid_offset_mask = pd.notna(merged_df['gmt_offset'])
    merged_df.loc[valid_offset_mask, 'local_time'] = merged_df.loc[valid_offset_mask, 'Timestamp'] + pd.to_timedelta(
        merged_df.loc[valid_offset_mask, 'gmt_offset'], unit='h')

    # 添加年、月、日、小时列
    merged_df['Year'] = merged_df['local_time'].dt.year
    merged_df['Month'] = merged_df['local_time'].dt.month
    merged_df['Day'] = merged_df['local_time'].dt.day
    merged_df['hour'] = merged_df['local_time'].dt.hour
    # 按日期和小时排序
    merged_df = merged_df.sort_values(by=['ROW', 'COL', 'Year', 'Month', 'Day', 'hour'])
    return merged_df


def save_daily_data_fusion_to_metrics(save_path, project_name, file_path, timezone_file):
    print("开始保存每日数据融合指标...")

    # 使用 dask 直接读取 CSV 文件
    dask_df = dd.read_csv(file_path, usecols=usecols, dtype=dtype)

    # 计算数据
    print("正在读取文件...")
    df = dask_df.compute()
    print("文件读取完毕。")

    timezone_df = pd.read_csv(timezone_file)

    # 将所有数据转换为本地时间
    df = convert_all_to_local_time(df, timezone_df)
    print("已完成时间转换为本地时间。")

    # 延迟计算
    mda8_task = calculate_mda8(df)

    # 并行计算MDA8
    mda8_df = compute(mda8_task)[0]

    other_metrics_task = calculate_other_metrics(mda8_df, save_path, project_name)

    output_file = compute(other_metrics_task)[0]
    print("每日数据融合指标保存完成.")
    return output_file


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Test/simulated_data.csv"
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly_True_Test.csv"

    # 读取时区偏移表
    timezone_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_ROWCOLRegion_Tz_CONUS_ST.csv'

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/Test"
    project_name = "2011_HourlyMetrics"

    # 调用函数计算指标并保存结果
    output_file = save_daily_data_fusion_to_metrics(save_path, project_name, file_path, timezone_file)
    print(f'指标文件已保存到: {output_file}')