import pandas as pd
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd
from dask import delayed, compute
import numba
# import calendar


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

    w126_metrics = []
    for col in ozone_columns:
        df_daytime[f'weighted_{col}'] = df_daytime[col] / (1 + 4403 * np.exp(-126 * df_daytime[col]))
        df_monthly = df_daytime.groupby(['Year', 'Month']).agg(
            {f'weighted_{col}': 'sum', col: 'count'}).reset_index()
        df_monthly.columns = ['year', 'month', f'monthly_w126_{col}', f'daytime_hours_count_{col}']
        # 根据每个月实际天数计算理论白天小时数（已注释掉），因为每小时的空间表吗数据理论上全有，除非每一小时的站点数全空，这不可能
        # df_monthly['total_possible_daytime_hours'] = df_monthly.apply(
        #     lambda row: calendar.monthrange(row['year'], row['month'])[1] * 12, axis=1)
        # df_monthly[f'available_ratio_{col}'] = df_monthly[f'daytime_hours_count_{col}'] / df_monthly['total_possible_daytime_hours']
        # df_monthly = df_monthly[df_monthly[f'available_ratio_{col}'] >= 0.75]
        # df_monthly[f'adjusted_monthly_w126_{col}'] = np.where(
        #     df_monthly[f'available_ratio_{col}'] < 1,
        #     df_monthly[f'monthly_w126_{col}'] * (1 / df_monthly[f'available_ratio_{col}']),
        #     df_monthly[f'monthly_w126_{col}']
        # )
        three_month_sums = []
        #滑窗
        for start_month in range(3, 9):
            end_month = start_month + 2
            subset = df_monthly[(df_monthly['month'] >= start_month) & (df_monthly['month'] <= end_month)]
            if len(subset) == 3:
                three_month_sum = subset[f'monthly_w126_{col}'].sum()
                three_month_sums.append(three_month_sum)
        if three_month_sums:
            w126_metric = max(three_month_sums)
            w126_metrics.append({'Period': f'W126_{col}', 'Value': w126_metric})

    w126_df = pd.DataFrame(w126_metrics)
    w126_df['ROW'] = -1
    w126_df['COL'] = -1
    for col in ozone_columns:
        w126_df[col] = np.nan
        w126_df.loc[w126_df['Period'] == f'W126_{col}', col] = w126_df[w126_df['Period'] == f'W126_{col}']['Value']

    w126_output_file = f'{save_path}/{project_name}_W126.csv'
    w126_df[['ROW', 'COL', 'model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'Period']].to_csv(w126_output_file, index=False)
    print(f"W126 中间结果数据已保存到: {w126_output_file}")

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
def calculate_mda8(df):
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
    print("MDA8 指标计算完成.")
    return pd.DataFrame(mda8_data)


def save_daily_data_fusion_to_metrics(save_path, project_name, file_path):
    print("开始保存每日数据融合指标...")

    # 使用 dask 直接读取 CSV 文件
    dask_df = dd.read_csv(file_path, usecols=usecols, dtype=dtype)

    # 计算数据
    df = dask_df.compute()
    print("文件读取完毕。")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 延迟计算
    mda8_task = calculate_mda8(df)
    w126_task = calculate_w126_metric(df, save_path, project_name)

    # 并行计算
    mda8_df, w126_metric_df = compute(mda8_task, w126_task)

    mda8_df['Year'] = mda8_df['Date'].apply(lambda x: x.year)
    mda8_df['Month'] = mda8_df['Date'].apply(lambda x: x.month)

    mda8_output_file = f'{save_path}/{project_name}_mda8.csv'
    mda8_df.to_csv(mda8_output_file, index=False)
    print(f"MDA8 数据已保存到: {mda8_output_file}")

    print(f"W126 指标数据已计算并准备处理.")

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
            all_metrics.append(agg_df)

    # 处理 W126 结果，将其转换为与其他指标一致的格式
    for _, row in w126_metric_df.iterrows():
        period = row['Period']
        value = row['Value']
        col = period.split('_')[1]
        # 假设这里 W126 指标没有 ROW 和 COL 信息，用 -1 填充
        w126_row = pd.DataFrame({
            'ROW': [-1],
            'COL': [-1],
            col: [value],
            'Period': [period]
        })
        all_metrics.append(w126_row)

    final_df = pd.concat(all_metrics, ignore_index=True)
    # 确保结果包含所有需要的列
    for col in ozone_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
    final_df = final_df[['ROW', 'COL'] + ozone_columns + ['Period']]
    output_file = f'{save_path}/{project_name}_metrics.csv'
    final_df.to_csv(output_file, index=False)
    print(f"最终指标数据已保存到: {output_file}")
    print("每日数据融合指标保存完成.")
    return [output_file, mda8_output_file]


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly.csv"

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011_Hourlymetrics_Optimze"

    # 调用函数计算指标并保存结果
    output_files = save_daily_data_fusion_to_metrics(save_path, project_name, file_path)
    print(f'指标文件已保存到: {output_files[0]}')
    print(f'MDA8 文件已保存到: {output_files[1]}')