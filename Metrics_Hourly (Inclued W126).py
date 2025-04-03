import pandas as pd
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd
from dask import delayed, compute
import numba


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

    # 按年和月分组并计算每月的加权和和计数
    unique_years_months, idx = np.unique(list(zip(years, months)), axis=0, return_index=True)
    monthly_weighted_sum = np.zeros(len(unique_years_months))
    for i, (year_month, start_idx) in enumerate(zip(unique_years_months, idx)):
        year, month = year_month
        mask = (years == year) & (months == month)
        monthly_weighted_sum[i] = np.sum(weighted_values[mask])

    # 滑窗计算
    window_size = 3
    three_month_sums = sliding_window_sum(monthly_weighted_sum, window_size)
    if len(three_month_sums) > 0:
        return np.max(three_month_sums)
    return np.nan


@delayed
def calculate_w126_metric(df_data, save_path, project_name):
    print("开始计算 W126 指标...")
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    # 转换单位，ppbv to ppm
    df_data[ozone_columns] = df_data[ozone_columns] / 1000
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month
    df_data['hour'] = df_data['Timestamp'].dt.hour
    df_daytime = df_data[(df_data['hour'] >= 8) & (df_data['hour'] < 20)]

    all_w126_metrics = []
    grouped = df_daytime.groupby(['ROW', 'COL'])
    for (row, col), group in tqdm(grouped, desc="Processing grids"):
        tasks = []
        for col_name in ozone_columns:
            task = calculate_single_w126(group, col_name)
            tasks.append(task)

        results = compute(*tasks)
        w126_metrics = {'ROW': row, 'COL': col, 'Period': 'W126'}
        for col_name, result in zip(ozone_columns, results):
            w126_metrics[col_name] = result

        all_w126_metrics.append(w126_metrics)

    w126_df = pd.DataFrame(all_w126_metrics)
    w126_output_file = f'{save_path}/{project_name}_W126.csv'
    w126_df[['ROW', 'COL', 'model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'Period']].to_csv(w126_output_file, index=False)
    print(f"W126 指标数据已保存到: {w126_output_file}")
    print("W126 指标计算完成.")
    return w126_df


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
def calculate_mda8(df, save_path, project_name):
    print("开始计算 MDA8 指标...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    mda8_data = []
    grouped = df.groupby(['ROW', 'COL', 'Date'])
    for (row, col, date), group in grouped:
        daily_mda8 = {'ROW': row, 'COL': col, 'Date': date}
        for col_name in ozone_columns:
            values = group[col_name].values
            rolling_means = rolling_mean_numba(values, 8)
            daily_mda8[col_name] = np.nanmax(rolling_means)
        mda8_data.append(daily_mda8)
    mda8_df = pd.DataFrame(mda8_data)
    # 将 MDA8 结果乘以 1000,因为并行计算前面W126会将df除1000影响这边，深拷贝太影响效率，乘1000就行
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    mda8_df[ozone_columns] = mda8_df[ozone_columns] * 1000
    mda8_output_file = f'{save_path}/{project_name}_mda8.csv'
    mda8_df['Year'] = mda8_df['Date'].apply(lambda x: x.year)
    mda8_df['Month'] = mda8_df['Date'].apply(lambda x: x.month)
    mda8_df.to_csv(mda8_output_file, index=False)
    print(f"MDA8 数据已保存到: {mda8_output_file}")
    print("MDA8 指标计算完成.")
    return mda8_df


@delayed
def calculate_other_metrics(mda8_df, save_path, project_name):
    print("开始计算其他指标（如DJF等）...")

    def top_10_average(series):
        return series.nlargest(10).mean()

    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    all_metrics = []
    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    periods = {
        'top-10': (top_10_average, ['ROW', 'COL'], mda8_df),
        'Annual': ('mean', ['ROW', 'COL', 'Year'], mda8_df),
        'Apr-Sep': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin([4, 5, 6, 7, 8, 9])]),
        'DJF': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['DJF'])]),
        'MAM': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['MAM'])]),
        'JJA': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['JJA'])]),
        'SON': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['SON'])])
    }
    for period, (aggregator, group_cols, temp_df) in periods.items():
        for col in ozone_columns:
            agg_df = temp_df.groupby(group_cols).agg({col: aggregator}).reset_index()
            agg_df['Period'] = f'{period}_{col}'
            for ozone_col in ozone_columns:
                if ozone_col not in agg_df.columns:
                    agg_df[ozone_col] = np.nan
            all_metrics.append(agg_df)

    final_df = pd.concat(all_metrics, ignore_index=True)
    final_df = final_df[['ROW', 'COL'] + ozone_columns + ['Period']]
    output_file = f'{save_path}/{project_name}_metrics.csv'
    final_df.to_csv(output_file, index=False)
    print(f"其他指标（如DJF等）数据已保存到: {output_file}")
    print("其他指标（如DJF等）计算完成.")
    return final_df


def save_daily_data_fusion_to_metrics(save_path, project_name, file_path):
    print("开始保存每日数据融合指标...")

    # 使用 dask 直接读取 CSV 文件
    dask_df = dd.read_csv(file_path, usecols=usecols, dtype=dtype)

    # 计算数据
    df = dask_df.compute()
    print("文件读取完毕。")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 延迟计算
    mda8_task = calculate_mda8(df, save_path, project_name)
    w126_task = calculate_w126_metric(df, save_path, project_name)

    # 并行计算MDA8和W126
    mda8_df, w126_df = compute(mda8_task, w126_task)

    other_metrics_task = calculate_other_metrics(mda8_df, save_path, project_name)
    # 并行计算其他指标和等待其他指标计算完成
    other_metrics_df = compute(other_metrics_task)[0]

    print("每日数据融合指标保存完成.")
    return [f'{save_path}/{project_name}_mda8.csv', f'{save_path}/{project_name}_W126.csv', f'{save_path}/{project_name}_metrics.csv']


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly.csv"

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011_HourlyMetrics"

    # 调用函数计算指标并保存结果
    output_files = save_daily_data_fusion_to_metrics(save_path, project_name, file_path)
    print(f'MDA8 文件已保存到: {output_files[0]}')
    print(f'W126 指标文件已保存到: {output_files[1]}')
    print(f'其他指标（如DJF等）文件已保存到: {output_files[2]}')
    