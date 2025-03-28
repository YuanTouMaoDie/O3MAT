import pandas as pd

# 读取第一个文件，提取 UniqueSite 和对应的 LatLon
file_path_1 = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv'
df_1 = pd.read_csv(file_path_1)
unique_sites = df_1.drop_duplicates(subset='Site')[['Site', 'Lat', 'Lon']]
unique_sites['Site'] = unique_sites['Site'].astype(str)
site_latlon_dict = unique_sites.set_index('Site')[['Lat', 'Lon']].to_dict(orient='index')

# 统计输入文件中的唯一站点个数
input_unique_site_count = unique_sites['Site'].nunique()

# 读取第二个文件，过滤出 site_id, POCode, dateon, O3 列并且 O3 != -999 的行
file_path_2 = '/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011.csv'
df_2 = pd.read_csv(file_path_2)
filtered_df = df_2[(df_2['O3'] != -999)][['site_id', 'POCode', 'dateon', 'O3']]
filtered_df['site_id'] = filtered_df['site_id'].astype(str)

# 添加经纬度信息
filtered_df['Lat'] = filtered_df['site_id'].map(lambda x: site_latlon_dict.get(x, {}).get('Lat'))
filtered_df['Lon'] = filtered_df['site_id'].map(lambda x: site_latlon_dict.get(x, {}).get('Lon'))

# 剔除不能添加上经纬度的行
filtered_df = filtered_df.dropna(subset=['Lat', 'Lon'])

# 统计输出文件中的唯一站点个数
output_unique_site_count = filtered_df['site_id'].nunique()

# 输出结果到指定文件
output_path = '/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011_LatLon.csv'
filtered_df.to_csv(output_path, index=False)

print(f"输入文件中的唯一站点个数: {input_unique_site_count}")
print(f"输出文件中的唯一站点个数: {output_unique_site_count}")
print(f"结果已成功保存到 {output_path}")
    