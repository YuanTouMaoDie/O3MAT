import pandas as pd


def print_unique_sites_count():
    file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_CV/2011_SixDataset_CV.csv"
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

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    for period_name, months in season_months.items():
        if months == 'all':
            subset = df
        else:
            subset = df[df['Month'].isin(months)]
        if not subset.empty:
            unique_sites_count = subset['Site'].nunique()
            print(f"Period: {period_name}, Unique Sites Count: {unique_sites_count}")

    for period_name, months in periods.items():
        if months == 'all':
            subset = df
        else:
            subset = df[df['Month'].isin(months)]
        if not subset.empty:
            unique_sites_count = subset['Site'].nunique()
            print(f"Period: {period_name}, Unique Sites Count: {unique_sites_count}")

    # 筛选出全年有至少 10 天数据的站点
    valid_sites = []
    for site, site_df in df.groupby('Site'):
        unique_dates = site_df['Date'].nunique()
        if unique_dates >= 10:
            valid_sites.append(site)

    top_10_df = df[df['Site'].isin(valid_sites)].sort_values(
        by='Conc', ascending=False).head(10)
    if not top_10_df.empty:
        unique_sites_count = top_10_df['Site'].nunique()
        print(f"Period: top - 10, Unique Sites Count: {unique_sites_count}")


print_unique_sites_count()
    