import pandas as pd
import numpy as np
from tqdm import tqdm
from dask import delayed, compute
import numba
from concurrent.futures import ThreadPoolExecutor


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


@numba.jit(nopython=True)
def calculate_weighted_values(values):
    return values / (1 + 4403 * np.exp(-126 * values))


@numba.jit(nopython=True)
def sliding_window_sum(arr, window_size):
    n = len(arr)
    result = np.empty(n - window_size + 1)
    for i in range(n - window_size + 1):
        result[i] = np.sum(arr[i:i + window_size])
    return result


@delayed
def calculate_single_w126(group, col_name):
    values = group[col_name].values
    weighted_values = calculate_weighted_values(values)
    months = group['Month'].values
    years = group['Year'].values

    # 按年和月分组并计算每月的加权和
    unique_years_months, idx = np.unique(list(zip(years, months)), axis=0, return_index=True)
    monthly_weighted_sum = np.zeros(len(unique_years_months))
    for i, (year_month, start_idx) in enumerate(zip(unique_years_months, idx)):
        year, month = year_month
        mask = (years == year) & (months == month)
        monthly_weighted_sum[i] = np.sum(weighted_values[mask])

    # 定义特定的 3 个月组合
    specific_month_combinations = [(3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10)]
    three_month_sums = []
    for start_month, middle_month, end_month in specific_month_combinations:
        start_mask = (unique_years_months[:, 1] == start_month)
        middle_mask = (unique_years_months[:, 1] == middle_month)
        end_mask = (unique_years_months[:, 1] == end_month)
        if np.sum(start_mask) > 0 and np.sum(middle_mask) > 0 and np.sum(end_mask) > 0:
            start_idx = np.where(start_mask)[0][0]
            middle_idx = np.where(middle_mask)[0][0]
            end_idx = np.where(end_mask)[0][0]
            three_month_sum = monthly_weighted_sum[start_idx] + monthly_weighted_sum[middle_idx] + monthly_weighted_sum[end_idx]
            three_month_sums.append(three_month_sum)

    if len(three_month_sums) > 0:
        return np.max(three_month_sums)
    return np.nan


def convert_single(row, df_data):
    df = df_data[(df_data['ROW'] == row['ROW']) & (df_data['COL'] == row['COL'])].copy()
    gmt_offset = row['gmt_offset']
    if pd.isna(gmt_offset):
        return pd.DataFrame()  # 返回一个空的 DataFrame，以便在后续 concat 中不引入错误数据
    # 使用 .loc 方法避免 SettingWithCopyWarning
    df.loc[:, 'Timestamp'] = pd.to_datetime(df['Timestamp'])
    local_time = df['Timestamp'] + pd.Timedelta(hours=gmt_offset)
    df['Year'] = local_time.dt.year
    df['Month'] = local_time.dt.month
    df['hour'] = local_time.dt.hour
    return df


def convert_all_to_local_time(df_data, timezone_df):
    """
    将所有数据的时间转换为本地时间，并添加年、月、小时列
    """
    all_local_data = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for _, row in timezone_df.iterrows():
            future = executor.submit(convert_single, row, df_data)
            futures.append(future)
        for future in futures:
            local_df = future.result()
            if not local_df.empty:  # 只添加非空的 DataFrame
                all_local_data.append(local_df)
    return pd.concat(all_local_data, ignore_index=True)


@delayed
def calculate_w126_metric(df_data, save_path, project_name):
    print("开始计算 W126 指标...")
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    # 转换单位，ppbv to ppm
    df_data[ozone_columns] = df_data[ozone_columns] / 1000

    # 根据localtime筛选
    df_2011 = df_data[(df_data['Year'] == 2011)]
    df_daytime = df_2011[(df_2011['hour'] >= 8) & (df_2011['hour'] < 20)]

    all_w126_metrics = []
    grouped = df_daytime.groupby(['ROW', 'COL'])
    for (row_num, col_num), group in grouped:
        tasks = []
        for col_name in ozone_columns:
            task = calculate_single_w126(group, col_name)
            tasks.append(task)

        results = compute(*tasks)
        w126_metrics = {'ROW': row_num, 'COL': col_num, 'Period': 'W126'}
        for col_name, result in zip(ozone_columns, results):
            w126_metrics[col_name] = result

        all_w126_metrics.append(w126_metrics)

    w126_df = pd.DataFrame(all_w126_metrics)
    w126_output_file = f'{save_path}/{project_name}_W126_4Hours.csv'
    w126_df[['ROW', 'COL', 'model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'Period']].to_csv(w126_output_file,
                                                                                                index=False)
    print(f"W126 指标数据已保存到: {w126_output_file}")
    print("W126 指标计算完成.")
    return w126_df


def save_w126_metrics(save_path, project_name, file_path, timezone_file):
    print("开始保存 W126 指标...")

    # 使用 dask 直接读取 CSV 文件
    dask_df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)
    timezone_df = pd.read_csv(timezone_file)

    print("文件读取完毕。")

    # 将所有数据转换为本地时间
    local_df = convert_all_to_local_time(dask_df, timezone_df)

    # 延迟计算
    w126_task = calculate_w126_metric(local_df, save_path, project_name)

    # 并行计算 W126
    w126_df = compute(w126_task)[0]

    print("W126 指标保存完成.")
    return f'{save_path}/{project_name}_W126_True.csv'


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly_True_Test.csv"

    # 读取时区偏移表
    timezone_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_ROWCOLRegion_Tz_CONUS.csv'

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011"

    # 调用函数计算指标并保存结果
    output_file = save_w126_metrics(save_path, project_name, file_path, timezone_file)
    print(f'W126 指标文件已保存到: {output_file}')
    