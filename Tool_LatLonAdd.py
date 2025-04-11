import pandas as pd
import re

# 读取第一个文件，提取 UniqueSite 和对应的 LatLon
file_path_1 = '/backupdata/data_EPA/EQUATES/EQUATES_data/SitesTable2001-2020.csv'
df_1 = pd.read_csv(file_path_1)
unique_sites = df_1.drop_duplicates(subset='Site')[['Site', 'Lat', 'Lon']]
unique_sites['Site'] = unique_sites['Site'].astype(str)
# 过滤掉带字母的站点
valid_sites = unique_sites[~unique_sites['Site'].str.contains(r'[a-zA-Z]', regex=True)]
site_latlon_dict = valid_sites.set_index('Site')[['Lat', 'Lon']].to_dict(orient='index')
# 找出被过滤掉的站点
filtered_out_sites_1 = set(unique_sites['Site']) - set(valid_sites['Site'])
print(f"从站点表中过滤掉的站点: {filtered_out_sites_1}")

# 站点表
unique_site_count = valid_sites['Site'].nunique()

# 读取第二个文件，只读取需要的列
file_path_2 = '/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011.csv'
try:
    df_2 = pd.read_csv(file_path_2, usecols=['site_id', 'POCode', 'dateon', 'O3'])
except Exception as e:
    print(f"读取文件时出现错误: {e}")

filtered_df = df_2[(df_2['O3'] != -999)]
filtered_df['site_id'] = filtered_df['site_id'].astype(str)
# 过滤掉带字母的站点
valid_input_sites = filtered_df[~filtered_df['site_id'].str.contains(r'[a-zA-Z]', regex=True)]
# 找出被过滤掉的站点
filtered_out_sites_2 = set(filtered_df['site_id']) - set(valid_input_sites['site_id'])
print(f"从输入文件中过滤掉的站点: {filtered_out_sites_2}")
# 统计输入文件中的唯一站点个数
input_unique_site_count = valid_input_sites['site_id'].nunique()

# 添加经纬度信息
valid_input_sites['Lat'] = valid_input_sites['site_id'].map(lambda x: site_latlon_dict.get(x, {}).get('Lat'))
valid_input_sites['Lon'] = valid_input_sites['site_id'].map(lambda x: site_latlon_dict.get(x, {}).get('Lon'))

# 剔除不能添加上经纬度的行
valid_input_sites = valid_input_sites.dropna(subset=['Lat', 'Lon'])

# 统计输出文件中的唯一站点个数
output_unique_site_count = valid_input_sites['site_id'].nunique()

# 输出结果到指定文件
output_path = '/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011_LatLon.csv'
valid_input_sites.to_csv(output_path, index=False)

print(f"站点表中的唯一站点个数: {unique_site_count}")
print(f"输入文件中O3站点的数据行数: {input_unique_site_count}")
print(f"输出文件中O3站点个数: {output_unique_site_count}")
print(f"结果已成功保存到 {output_path}")