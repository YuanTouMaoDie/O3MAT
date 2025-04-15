import pandas as pd
import numpy as np
import numba
from tqdm import tqdm

# 定义需要读取的列
usecols = ['ROW', 'COL', 'Timestamp', 'vna_ozone', 'avna_ozone', 'evna_ozone', 'model']

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

# 使用numba优化滑窗求和函数
@numba.jit(nopython=True)
def sliding_window_sum_numba(arr, window_size):
    n = len(arr)
    result = np.empty(n - window_size + 1, dtype=arr.dtype)
    window_sum = np.sum(arr[:window_size])
    result[0] = window_sum
    for i in range(1, n - window_size + 1):
        window_sum = window_sum - arr[i - 1] + arr[i + window_size - 1]
        result[i] = window_sum
    return result


def calculate_mda8o3(group, ozone_columns):
    mda8_values = {}
    for col in ozone_columns:
        values = group[col].values
        if len(values) >= 8:
            # 使用优化后的滑窗求和函数
            window_sums = sliding_window_sum_numba(values, 8)
            mda8 = np.max(window_sums / 8)
        else:
            mda8 = np.nan
        mda8_values[col] = mda8
    return mda8_values


def calculate_top_10_average(series):
    return series.nlargest(10).mean()


def calculate_metrics_from_mda8(mda8_df):
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    all_metrics = []
    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    periods = {
        'top-10': (calculate_top_10_average, ['ROW', 'COL']),
        'Annual': ('mean', ['ROW', 'COL', 'Year']),
        'Apr-Sep': ('mean', ['ROW', 'COL']),
        'DJF': ('mean', ['ROW', 'COL']),
        'MAM': ('mean', ['ROW', 'COL']),
        'JJA': ('mean', ['ROW', 'COL']),
        'SON': ('mean', ['ROW', 'COL'])
    }

    # 预处理分组和聚合操作
    groupby_results = {}
    for period, (agg_func, groupby_cols) in periods.items():
        if agg_func == 'mean':
            groupby_result = mda8_df.groupby(groupby_cols)[ozone_columns].mean()
        else:
            groupby_result = mda8_df.groupby(groupby_cols)[ozone_columns].apply(lambda x: x.apply(agg_func))
        groupby_result = groupby_result.reset_index()
        groupby_results[period] = groupby_result

    # 生成最终结果
    for period, groupby_result in groupby_results.items():
        for col in ozone_columns:
            result_df = groupby_result[['ROW', 'COL'] + [col]].copy()
            result_df['Period'] = f'{period}_{col}'
            all_metrics.append(result_df)

    final_metrics = pd.concat(all_metrics, axis=0)

    columns_order = ['ROW', 'COL', 'Period'] + ozone_columns
    result_df = final_metrics[columns_order]

    return result_df


def process_grid(grid_data, df_2011):
    row, col = grid_data
    # 使用loc方法避免数据复制
    group = df_2011.loc[(df_2011['ROW'] == row) & (df_2011['COL'] == col)]
    mda8_values = calculate_mda8o3(group, ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model'])
    mda8_values.update({'ROW': row, 'COL': col, 'Month': group['Month'].iloc[0], 'Year': group['Year'].iloc[0]})  # 添加 Month 列
    return pd.DataFrame([mda8_values])


def save_metrics(save_path, project_name, file_path, timezone_file):
    print("开始保存指标...")

    # 读取 CSV 文件
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)
    timezone_df = pd.read_csv(timezone_file)

    print("文件读取完毕。")

    # 将所有数据转换为本地时间
    local_df = convert_all_to_local_time(df, timezone_df)
    print("已完成时间转换为本地时间。")

    # 筛选 2011 年数据
    df_2011 = local_df[local_df['Year'] == 2011]

    # 提前过滤缺失值
    df_2011 = df_2011.dropna(subset=['Timestamp'])

    # 获取所有不同的网格组合
    grid_combinations = df_2011[['ROW', 'COL']].drop_duplicates().values

    results = []
    for grid in tqdm(grid_combinations, desc="处理网格进度"):
        results.append(process_grid(grid, df_2011))

    # 将结果合并成一个DataFrame
    mda8_df = pd.concat(results, axis=0)

    # 计算新指标
    metrics_df = calculate_metrics_from_mda8(mda8_df)

    metrics_output_file = f'{save_path}/{project_name}_MDA8_metrics_1.csv'
    metrics_df.to_csv(metrics_output_file, index=False)
    print(f"MDA8 指标数据已保存到: {metrics_output_file}")

    print("指标保存完成.")
    return metrics_output_file


def convert_all_to_local_time(df_data, timezone_df):
    """
    将所有数据的时间转换为本地时间，并添加年、月、日、小时列
    """

    # 合并数据
    merged_df = pd.merge(df_data, timezone_df, on=['ROW', 'COL'], how='left')
    # 转换时间戳
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])

    # 使用向量化操作计算本地时间
    valid_offset_mask = pd.notna(merged_df['gmt_offset'])
    merged_df.loc[valid_offset_mask, 'local_time'] = merged_df.loc[valid_offset_mask, 'Timestamp'] + pd.to_timedelta(merged_df.loc[valid_offset_mask, 'gmt_offset'], unit='h')

    # 添加年、月、日、小时列
    merged_df['Year'] = merged_df['local_time'].dt.year
    merged_df['Month'] = merged_df['local_time'].dt.month
    merged_df['Day'] = merged_df['local_time'].dt.day
    merged_df['hour'] = merged_df['local_time'].dt.hour
    # 移除不必要的排序
    # merged_df = merged_df.sort_values(by=['ROW', 'COL', 'Year', 'Month', 'Day', 'hour'])
    return merged_df


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    # file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly_ST.csv"
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/Test/simulated_data.csv"

    # 读取时区偏移表
    timezone_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_ROWCOLRegion_Tz_CONUS_ST.csv'

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011"

    # 调用函数计算指标并保存结果
    output_file = save_metrics(save_path, project_name, file_path, timezone_file)
    print(f'指标文件已保存到: {output_file}')