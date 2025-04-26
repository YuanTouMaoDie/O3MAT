import pandas as pd

# 初始化一个空字典，用于存储所有站点经纬度信息
all_site_latlon_dict = {}

# 遍历2001到2019年
for year in range(2001, 2020):
    # 构建文件路径
    file_path = f'/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.{year}.csv'
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 提取唯一站点及其经纬度
        unique_sites = df.drop_duplicates(subset='Site')[['Site', 'Lat', 'Lon']]
        # 将站点ID转换为字符串类型
        unique_sites['Site'] = unique_sites['Site'].astype(str)
        # 遍历每个唯一站点，更新总字典
        for _, row in unique_sites.iterrows():
            site = row['Site']
            lat = row['Lat']
            lon = row['Lon']
            all_site_latlon_dict[site] = {'Lat': lat, 'Lon': lon}
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件 {file_path} 时出现错误: {e}")

# 处理新加入的文件
new_files = [
    '/backupdata/data_EPA/EQUATES/EQUATES_data/SMAT_OZONE_MDA1_APRSEP_STD70_2000_2022.CSV',
    '/backupdata/data_EPA/EQUATES/EQUATES_data/SMAT_OZONE_MDA8_MAYSEP_STD70_2000_2022.CSV'
]

for file_path in new_files:
    try:
        df = pd.read_csv(file_path)
        # 假设新文件里的站点和经纬度列名和之前的一样，如果不同需修改
        unique_sites = df.drop_duplicates(subset='Site')[['Site', 'Lat', 'Lon']]
        unique_sites['Site'] = unique_sites['Site'].astype(str)
        for _, row in unique_sites.iterrows():
            site = row['Site']
            lat = row['Lat']
            lon = row['Lon']
            all_site_latlon_dict[site] = {'Lat': lat, 'Lon': lon}
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件 {file_path} 时出现错误: {e}")

# 处理 MonitorsTimeRegion_Filter.csv 文件
monitors_file = '/output/Region/MonitorsTimeRegion_Filter.csv'
try:
    df = pd.read_csv(monitors_file)
    if 'site_id' in df.columns and 'Lat' in df.columns and 'Lon' in df.columns:
        unique_sites = df.drop_duplicates(subset='site_id')[['site_id', 'Lat', 'Lon']]
        unique_sites['site_id'] = unique_sites['site_id'].astype(str)
        for _, row in unique_sites.iterrows():
            site = row['site_id']
            lat = row['Lat']
            lon = row['Lon']
            all_site_latlon_dict[site] = {'Lat': lat, 'Lon': lon}
    else:
        print(f"文件 {monitors_file} 中缺少必要的列（site_id, Lat, Lon）。")
except FileNotFoundError:
    print(f"文件 {monitors_file} 未找到。")
except Exception as e:
    print(f"读取文件 {monitors_file} 时出现错误: {e}")

# 将汇总的字典转换为DataFrame
result_df = pd.DataFrame.from_dict(all_site_latlon_dict, orient='index')
result_df.index.name = 'Site'

# 保存结果到CSV文件
output_path = '/backupdata/data_EPA/EQUATES/EQUATES_data/SitesTable2001-2020.csv'
try:
    result_df.to_csv(output_path)
    print(f"结果已成功保存到 {output_path}")
except Exception as e:
    print(f"保存文件时出现错误: {e}")

# 输出结果数据表中唯一站点的信息
unique_sites = result_df.index.unique()
print("输出数据表中的唯一站点:")
print(unique_sites)