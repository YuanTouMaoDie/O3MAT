import pandas as pd


def calculate_metrics():
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2018_Data_CV/2018_SixDataset_CV.csv"
    df = pd.read_csv(file_path)

    season_months = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    periods = {
        'Annual': 'all',
        'Apr-Sep': [4, 5, 6, 7, 8, 9]
    }
    all_metrics = []
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    for site, site_df in df.groupby('Site'):
        for period_name, months in season_months.items():
            subset = site_df[site_df['Month'].isin(months)]
            if not subset.empty:
                mean_data = subset[['Lat', 'Lon', 'Conc', 'model', 'vna_ozone', 'evna_ozone',
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
                mean_data = subset[['Lat', 'Lon', 'Conc', 'model', 'vna_ozone', 'evna_ozone',
                                    'avna_ozone', 'ds_ozone', 'ROW', 'COL']].mean().reset_index()
                mean_data.columns = ['Metric', 'Value']
                mean_data['Period'] = period_name
                mean_data['Site'] = site
                pivot_data = mean_data.pivot(index=['Site', 'Period'], columns='Metric', values='Value').reset_index()
                all_metrics.append(pivot_data)

        sorted_df = site_df.sort_values(by='Conc', ascending=False)
        top_10_df = sorted_df.head(10)
        top_10_mean = top_10_df[['Lat', 'Lon', 'Conc', 'model', 'vna_ozone', 'evna_ozone',
                                 'avna_ozone', 'ds_ozone', 'ROW', 'COL']].mean().reset_index()
        top_10_mean.columns = ['Metric', 'Value']
        top_10_mean['Period'] = 'top-10'
        top_10_mean['Site'] = site
        pivot_data = top_10_mean.pivot(index=['Site', 'Period'], columns='Metric', values='Value').reset_index()
        all_metrics.append(pivot_data)

    result = pd.concat(all_metrics, ignore_index=True)
    result = result[['Site', 'Lat', 'Lon', 'Conc', 'model', 'vna_ozone', 'evna_ozone',
                     'avna_ozone', 'ds_ozone', 'ROW', 'COL', 'Period']]
    return result


result = calculate_metrics()
# 保存结果到 CSV 文件
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2018_Data_CV/2018_SixDataset_CV_Metrics.csv'
result.to_csv(output_file, index=False)
print(f"结果已保存到 {output_file}")