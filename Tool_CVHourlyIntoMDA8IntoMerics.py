import pandas as pd


def calculate_metrics():
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_CV/2011_SixDataset_CV_hourly.csv"
    df = pd.read_csv(file_path)

    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    periods = {
        'Annual': 'all',
        'Apr - Sep': [4, 5, 6, 7, 8, 9]
    }
    all_metrics = []

    # 将 dateon 列转换为日期时间类型
    df['Date'] = pd.to_datetime(df['dateon'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour

    # 按站点和日期分组计算每天的 MDA8
    columns_to_calculate = ['O3', 'vna_ozone', 'evna_ozone', 'avna_ozone']
    mda8_df = df.groupby(['site_id', df['Date'].dt.date]).apply(lambda x: calculate_daily_mda8(x, columns_to_calculate)).reset_index(drop=True)
    mda8_df['Month'] = pd.to_datetime(mda8_df['Date']).dt.month

    for site, site_df in mda8_df.groupby('site_id'):
        for period_name, months in season_months.items():
            subset = site_df[site_df['Month'].isin(months)]
            if not subset.empty:
                mean_data = subset[['Lat', 'Lon', 'MDA8O3', 'MDA8_vna_ozone', 'MDA8_evna_ozone', 'MDA8_avna_ozone', 'CVgroup','model', 'vna_ozone', 'evna_ozone',
                                    'avna_ozone', 'ds_ozone', 'ROW', 'COL']].mean().reset_index()
                mean_data.columns = ['Metric', 'Value']
                mean_data['Period'] = period_name
                mean_data['Site'] = site
                pivot_data = mean_data.pivot(index=['Site', 'Period'], columns='Metric', values='Value').reset_index()
                all_metrics.append(pivot_data)

        for period_name, months in periods.items():
            if months == 'all':
                subset = site_df
            else:
                subset = site_df[site_df['Month'].isin(months)]
            if not subset.empty:
                mean_data = subset[['Lat', 'Lon', 'MDA8O3', 'MDA8_vna_ozone', 'MDA8_evna_ozone', 'MDA8_avna_ozone', 'CVgroup','model', 'vna_ozone', 'evna_ozone',
                                    'avna_ozone', 'ds_ozone', 'ROW', 'COL']].mean().reset_index()
                mean_data.columns = ['Metric', 'Value']
                mean_data['Period'] = period_name
                mean_data['Site'] = site
                pivot_data = mean_data.pivot(index=['Site', 'Period'], columns='Metric', values='Value').reset_index()
                all_metrics.append(pivot_data)

        sorted_df = site_df.sort_values(by='MDA8O3', ascending=False)
        top_10_df = sorted_df.head(10)
        top_10_mean = top_10_df[['Lat', 'Lon', 'MDA8O3', 'MDA8_vna_ozone', 'MDA8_evna_ozone', 'MDA8_avna_ozone', 'CVgroup','model', 'vna_ozone', 'evna_ozone',
                                 'avna_ozone', 'ds_ozone', 'ROW', 'COL']].mean().reset_index()
        top_10_mean.columns = ['Metric', 'Value']
        top_10_mean['Period'] = 'top - 10'
        top_10_mean['Site'] = site
        pivot_data = top_10_mean.pivot(index=['Site', 'Period'], columns='Metric', values='Value').reset_index()
        all_metrics.append(pivot_data)

    result = pd.concat(all_metrics, ignore_index=True)
    result = result[['Site', 'Lat', 'Lon', 'MDA8O3', 'MDA8_vna_ozone', 'MDA8_evna_ozone', 'MDA8_avna_ozone', 'CVgroup','model', 'vna_ozone', 'evna_ozone',
                     'avna_ozone', 'ds_ozone', 'ROW', 'COL', 'Period']]
    return result


def calculate_daily_mda8(group, columns):
    # 按小时排序
    group = group.sort_values(by='Hour')
    first_row = group.iloc[0].copy()
    for col in columns:
        # 计算 8 小时滑动平均
        rolling_mean = group[col].rolling(window=8, min_periods=8).mean()
        # 取最大值作为当天的 MDA8
        mda8 = rolling_mean.max()
        first_row[f'MDA8_{col}'] = mda8
    return first_row


result = calculate_metrics()
# 保存结果到 CSV 文件
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_CV/2011_SixDataset_CV_hourlyIntoMDA8O3_Metrics.csv'
result.to_csv(output_file, index=False)
print(f"结果已保存到 {output_file}")