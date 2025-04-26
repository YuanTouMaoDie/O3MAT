import pandas as pd
import numpy as np
import numba
from joblib import Parallel, delayed
from scipy.signal import convolve


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


# 使用卷积操作实现滑窗求和
def sliding_window_sum_convolve(arr, window_size):
    kernel = np.ones(window_size, dtype=arr.dtype)
    result = convolve(arr, kernel, mode='valid')
    return result


def convert_all_to_local_time(df_data, timezone_df):
    """
    将所有数据的时间转换为本地时间，并添加年、月、日、小时列
    """
    # 合并数据
    merged_df = pd.merge(df_data, timezone_df, on=['ROW', 'COL'], how='left')
    # 转换时间戳
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
    # 根据各个网格所在的时区将UTC转为相应的localtime, like ST = UTC + -5
    merged_df['local_time'] = merged_df['Timestamp'] + pd.to_timedelta(merged_df['gmt_offset'], unit='h')
    # 添加年、月、日、小时列
    merged_df['Year'] = merged_df['local_time'].dt.year
    merged_df['Month'] = merged_df['local_time'].dt.month
    merged_df['Day'] = merged_df['local_time'].dt.day
    merged_df['hour'] = merged_df['local_time'].dt.hour
    # 按日期和小时排序
    merged_df = merged_df.sort_values(by=['ROW', 'COL', 'Year', 'Month', 'Day', 'hour'])
    return merged_df


def calculate_mda8o3(group, ozone_columns):
    mda8_values = {}
    for col in ozone_columns:
        values = group[col].values
        if len(values) >= 8:
            # 使用优化后的滑窗求和函数
            mda8 = np.max(sliding_window_sum_convolve(values, 8) / 8)
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
        'top-10': (calculate_top_10_average, ['ROW', 'COL'], mda8_df),
        'Annual': ('mean', ['ROW', 'COL', 'Year'], mda8_df),
        'Apr-Sep': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin([4, 5, 6, 7, 8, 9])]),
        'DJF': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['DJF'])]),
        'MAM': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['MAM'])]),
        'JJA': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['JJA'])]),
        'SON': ('mean', ['ROW', 'COL'], mda8_df[mda8_df['Month'].isin(season_months['SON'])])
    }

    for period, (agg_func, groupby_cols, df) in periods.items():
        grouped = df.groupby(groupby_cols)
        for col in ozone_columns:
            if isinstance(agg_func, str):
                result = grouped[col].agg(agg_func)
            else:
                result = grouped[col].apply(agg_func)
            result_df = result.reset_index()
            result_df['Period'] = period
            result_df.rename(columns={col: f'{col}_{period}'}, inplace=True)
            all_metrics.append(result_df)

    final_metrics = pd.concat(all_metrics, axis=0)

    # 数据透视转换为宽格式
    id_vars = ['ROW', 'COL', 'Period']
    var_name = 'OzoneType_Period'
    value_name = 'Value'
    melted = final_metrics.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
    melted[['OzoneType', 'Period']] = melted[var_name].str.split('_', expand=True)
    pivoted = melted.pivot(index=['ROW', 'COL', 'Period'], columns='OzoneType', values='Value').reset_index()
    pivoted.columns.name = None

    return pivoted


def process_grid(grid_data, df_2011):
    row, col = grid_data
    group = df_2011[(df_2011['ROW'] == row) & (df_2011['COL'] == col)]
    mda8_values = calculate_mda8o3(group, ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model'])
    mda8_values.update({'ROW': row, 'COL': col})
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

    # 获取所有不同的网格组合
    grid_combinations = df_2011[['ROW', 'COL']].drop_duplicates().values

    # 并行处理每个网格
    results = Parallel(n_jobs=-1)(delayed(process_grid)(grid, df_2011) for grid in grid_combinations)

    # 将结果合并成一个DataFrame
    mda8_df = pd.concat(results, axis=0)

    # 计算新指标
    metrics_df = calculate_metrics_from_mda8(mda8_df)

    metrics_output_file = f'{save_path}/{project_name}_MDA8_metrics_Exsame.csv'
    metrics_df.to_csv(metrics_output_file, index=False)
    print(f"MDA8 指标数据已保存到: {metrics_output_file}")

    print("指标保存完成.")
    return metrics_output_file


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_Data_HourlySTExsame.csv"

    # 读取时区偏移表
    timezone_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/ROWCOLRegion_Tz_CONUS_ST.csv'

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011"

    # 调用函数计算指标并保存结果
    output_file = save_metrics(save_path, project_name, file_path, timezone_file)
    print(f'指标文件已保存到: {output_file}')