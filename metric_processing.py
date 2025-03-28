import pandas as pd
import numpy as np


def calculate_w126_metric(df_data):
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model', 'harvard_ml', 'ds_ozone']
    # 转换单位
    for col in ozone_columns:
        df_data[col] = df_data[col] / 1000
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month
    df_data['hour'] = df_data['Timestamp'].dt.hour
    df_daytime = df_data[(df_data['hour'] >= 8) & (df_data['hour'] <= 20)]
    w126_metrics = []
    for col in ozone_columns:
        df_daytime[f'weighted_{col}'] = df_daytime[col] / (1 + 4403 * np.exp(-126 * df_daytime[col]))
        df_monthly = df_daytime.groupby([df_daytime['Year'], df_daytime['Month']]).agg(
            {f'weighted_{col}': 'sum', col: 'count'}).reset_index()
        df_monthly.columns = ['year', 'month', f'monthly_w126_{col}', f'daytime_hours_count_{col}']
        total_possible_daytime_hours = 31 * 12
        df_monthly[f'available_ratio_{col}'] = df_monthly[f'daytime_hours_count_{col}'] / total_possible_daytime_hours
        df_monthly = df_monthly[df_monthly[f'available_ratio_{col}'] >= 0.75]
        df_monthly[f'adjusted_monthly_w126_{col}'] = df_monthly.apply(
            lambda row: row[f'monthly_w126_{col}'] * (1 / row[f'available_ratio_{col}'])
            if row[f'available_ratio_{col}'] < 1 else row[f'monthly_w126_{col}'],
            axis=1
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
    return pd.DataFrame(w126_metrics)


def calculate_mda8(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model', 'harvard_ml', 'ds_ozone']
    mda8_data = []
    for (row, col, date), group in df.groupby(['ROW', 'COL', 'Date']):
        daily_mda8 = {'ROW': row, 'COL': col, 'Date': date}
        for col_name in ozone_columns:
            mda8 = group[col_name].rolling(window=8).mean().max()
            daily_mda8[col_name] = mda8
        mda8_data.append(daily_mda8)
    return pd.DataFrame(mda8_data)


def save_daily_data_fusion_to_metrics(df_data, save_path, project_name):
    mda8_df = calculate_mda8(df_data)
    mda8_output_file = f'{save_path}/{project_name}_mda8.csv'
    mda8_df.to_csv(mda8_output_file, index=False)

    w126_metric_df = calculate_w126_metric(df_data)

    def top_10_average(series):
        return series.nlargest(10).mean()

    ozone_columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model', 'harvard_ml', 'ds_ozone']
    all_metrics = []
    df_data['Year'] = pd.to_datetime(df_data['Timestamp']).dt.year
    df_data['Month'] = pd.to_datetime(df_data['Timestamp']).dt.month
    for period in ['top-10', 'Annual', 'Apr-Sep', 'DJF', 'MAM', 'JJA', 'SON']:
        if period == 'top-10':
            aggregator = top_10_average
            group_cols = ['ROW', 'COL']
            temp_df = df_data
        elif period == 'Annual':
            aggregator = 'mean'
            group_cols = ['ROW', 'COL', 'Year']
            temp_df = df_data
        elif period == 'Apr-Sep':
            aggregator = 'mean'
            group_cols = ['ROW', 'COL']
            temp_df = df_data[df_data['Month'].isin([4, 5, 6, 7, 8, 9])]
        elif period in ['DJF', 'MAM', 'JJA', 'SON']:
            aggregator = 'mean'
            group_cols = ['ROW', 'COL']
            season_months = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11]
            }
            temp_df = df_data[df_data['Month'].isin(season_months[period])]
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
    return [output_file, mda8_output_file]

if __name__ == "__main__":
    # 读取输入文件
    file_path = "2011_SixDataset_Hourly.csv"
    df = pd.read_csv(file_path)

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    project_name = "2011_metrics"

    # 调用函数计算指标并保存结果
    output_files = save_daily_data_fusion_to_metrics(df, save_path, project_name)
    print(f'指标文件已保存到: {output_files[0]}')
    print(f'MDA8 文件已保存到: {output_files[1]}')
    