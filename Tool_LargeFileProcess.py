import pandas as pd
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd

# 定义需要读取的列
usecols = ['ROW', 'COL', 'Timestamp', 'vna_ozone', 'avna_ozone', 'evna_ozone', 'model']

# 定义每列的数据类型
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
def calculate_w126_metric(df_data):
    print("开始计算 W126 指标...")
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    # 转换单位
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
        total_possible_daytime_hours = 31 * 12
        df_monthly[f'available_ratio_{col}'] = df_monthly[f'daytime_hours_count_{col}'] / total_possible_daytime_hours
        df_monthly = df_monthly[df_monthly[f'available_ratio_{col}'] >= 0.75]
        df_monthly[f'adjusted_monthly_w126_{col}'] = np.where(
            df_monthly[f'available_ratio_{col}'] < 1,
            df_monthly[f'monthly_w126_{col}'] * (1 / df_monthly[f'available_ratio_{col}']),
            df_monthly[f'monthly_w126_{col}']
        )
        three_month_sums = []
        for start_month in range(3, 9):
            end_month = start_month + 2
            subset = df_monthly[(df_monthly['month'] >= start_month) & (df_monthly['month'] <= end_month)]
            if len(subset) == 3:
                three_month_sum = subset[f'adjusted_monthly_w126_{col}'].sum()
                three_month_sums.append(three_month_sum)
        if three_month_sums:
            w126_metric = max(three_month_sums)
            w126_metrics.append({'Period': f'W126_{col}', 'Value': w126_metric})
    print("W126 指标计算完成.")
    return pd.DataFrame(w126_metrics)

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
            daily_mda8[col_name] = group[col_name].rolling(window=8).mean().max()
        mda8_data.append(daily_mda8)
    print("MDA8 指标计算完成.")
    return pd.DataFrame(mda8_data)


def save_daily_data_fusion_to_metrics(save_path, project_name, file_path):
    print("开始保存每日数据融合指标...")

    # 使用 dask 并行读取 CSV 文件
    dask_df = dd.read_csv(file_path, usecols=usecols, dtype=dtype)

    # 获取分块数量
    total_chunks = len(dask_df.divisions)

    df_parts = []

    # 使用 tqdm 显示进度条
    with tqdm(total=total_chunks, desc="Reading CSV file") as pbar:
        for i in range(total_chunks - 1):  # 修改循环范围
            chunk = dask_df.partitions[i].compute()
            df_parts.append(chunk)
            pbar.update(1)

    # 处理最后一个分区
    if total_chunks > 0:
        last_chunk = dask_df.partitions[total_chunks - 1].compute()
        df_parts.append(last_chunk)
        pbar.update(1)

    df = pd.concat(df_parts, ignore_index=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    mda8_df = calculate_mda8(df)
    mda8_df['Year'] = mda8_df['Date'].apply(lambda x: x.year)
    mda8_df['Month'] = mda8_df['Date'].apply(lambda x: x.month)

    mda8_output_file = f'{save_path}/{project_name}_mda8.csv'
    mda8_df.to_csv(mda8_output_file, index=False)
    print(f"MDA8 数据已保存到: {mda8_output_file}")

    w126_metric_df = calculate_w126_metric(df)
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
    project_name = "2011_Hourlymetrics"

    # 调用函数计算指标并保存结果
    output_files = save_daily_data_fusion_to_metrics(save_path, project_name, file_path)
    print(f'指标文件已保存到: {output_files[0]}')
    print(f'MDA8 文件已保存到: {output_files[1]}')
    