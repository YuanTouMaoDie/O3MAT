import pandas as pd

# 加载数据
file_path = "/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
data = pd.read_csv(file_path)

# 将 'Date' 列转换为日期时间格式
data['Date'] = pd.to_datetime(data['Date'])

# 定义季度函数
def get_quarter(month):
    if month in [12, 1, 2]:
        return 'DFJ'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'

# 添加季度列
data['Quarter'] = data['Date'].dt.month.apply(get_quarter)

# 生成每个季度的日期范围
quarters = {
    'DJF': pd.date_range('2011-01-01', '2011-02-28').union(pd.date_range('2011-12-01', '2011-12-31')),
    'MAM': pd.date_range('2011-03-01', '2011-05-31'),
    'JJA': pd.date_range('2011-06-01', '2011-08-31'),
    'SON': pd.date_range('2011-09-01', '2011-11-30'),
    'Apr-Sep': pd.date_range('2011-04-01', '2011-09-30'),
    'Annual': pd.date_range('2011-01-01', '2011-12-31'),
    'top-10': pd.date_range('2011-01-01', '2011-12-31'),
    '98th': pd.date_range('2011-01-01', '2011-12-31'),
}

# 计算缺失日期的数量
def count_missing_dates(site_data, quarter_dates):
    site_quarter_data = site_data[site_data['Date'].isin(quarter_dates)]
    missing_dates = len(quarter_dates) - len(site_quarter_data)
    return missing_dates

# 统计每个站点各个季度的缺失天数
missing_days_per_site = []

for site in data['Site'].unique():
    site_data = data[data['Site'] == site]
    lat = site_data['Lat'].iloc[0]  # 获取站点纬度
    lon = site_data['Lon'].iloc[0]  # 获取站点经度

    for quarter, quarter_dates in quarters.items():
        missing_days = count_missing_dates(site_data, quarter_dates)
        missing_days_per_site.append([site, lon, lat, missing_days, quarter])

# 创建 DataFrame
missing_days_df = pd.DataFrame(missing_days_per_site, columns=['StationID', 'Lon', 'Lat', 'Conc', 'Period'])

# ...existing code...

# 保存结果到 CSV 文件
output_file_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/missing_days_per_site_2011.csv"
missing_days_df.to_csv(output_file_path, index=False)
