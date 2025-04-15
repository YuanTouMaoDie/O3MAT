import pandas as pd

# #提取出开头8的站点
# def extract_site_latlon(csv_file, output_csv):
#     try:
#         # 读取 CSV 文件
#         df = pd.read_csv(csv_file)

#         # 筛选 Site 为 8 位数字且开头为 8 的行
#         filtered_df = df[df['Site'].astype(str).str.match(r'^8\d{8}$')]

#         # 提取 Site、Lat 和 Lon 列
#         result_df = filtered_df[['Site', 'Lat', 'Lon']]

#         # 保存到 CSV 文件
#         result_df.to_csv(output_csv, index=False)
#         print(f"数据已成功保存到 {output_csv}")

#     except FileNotFoundError:
#         print("错误: 文件未找到，请检查文件路径。")
#     except KeyError:
#         print("错误: 数据表中没有 'Site'、'Lat' 或 'Lon' 列。")
#     except Exception as e:
#         print(f"发生未知错误: {e}")


# if __name__ == "__main__":
#     csv_file = '/backupdata/data_EPA/EQUATES/EQUATES_data/SitesTable2001-2020.csv'
#     output_csv = '/backupdata/data_EPA/EQUATES/EQUATES_data/SitesTable2001-2020_8.csv'
#     extract_site_latlon(csv_file, output_csv)

# #画站点图
# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# from shapely.geometry import Point

# # 读取 GeoJSON 文件
# us_map = gpd.read_file('output/Region/USA_State.json')

# # 筛选出美国本土州的数据，排除阿拉斯加和夏威夷
# us_continental_map = us_map[~us_map['name'].isin(['Alaska', 'Hawaii'])]

# # 读取包含数据点的 CSV 文件
# data = pd.read_csv('/backupdata/data_EPA/EQUATES/EQUATES_data/SitesTable2001-2020_8.csv')

# # 将数据点转换为 GeoDataFrame
# points = gpd.GeoDataFrame(
#     data,
#     geometry=gpd.points_from_xy(data['Lon'], data['Lat']),
#     crs=us_continental_map.crs
# )

# # 创建绘图对象
# fig, ax = plt.subplots(figsize=(10, 8))

# # 绘制美国本土地图，设置填充颜色和边界颜色及样式
# us_continental_map.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.5)

# # 在地图上绘制数据点，设置颜色、大小和透明度
# points.plot(ax=ax, color='Crimson', markersize=6, alpha=0.7)

# # 设置坐标轴范围以显示左下角区域，你可以根据实际情况调整
# xlim = (-125, -65)
# ylim = (20, 50)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

# # 添加标题
# ax.set_title('US Continental Lower Left Region with Data Points', fontsize=16)

# # 设置坐标轴标签
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# # 设置坐标轴刻度
# xticks = range(-125, -65, 5)
# yticks = range(20, 50, 1)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)

# # 显示网格线
# ax.grid(True, linestyle='--', alpha=0.5)

# # 显示图形
# plt.show()
#  
#画出不同点不同时区
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.cm as cm
import numpy as np

# 读取美国地图的 GeoJSON 文件
us_map = gpd.read_file('output/Region/USA_State.json')

# 筛选出美国本土州的数据，排除阿拉斯加和夏威夷
us_continental_map = us_map[~us_map['name'].isin(['Alaska', 'Hawaii'])]

# 进一步通过经纬度范围筛选（示例范围，可按需调整）
min_lon = -125
max_lon = -65
min_lat = 25
max_lat = 50
us_continental_map = us_continental_map.cx[min_lon:max_lon, min_lat:max_lat]

# 读取包含监测点信息的 CSV 文件
data = pd.read_csv('output/Region/MonitorsTimeRegion_Filter_ST_QA_Compare.csv')

# 过滤 Lat 大于 24 且小于 50 的数据
filtered_data = data[(data['Lat'] > 24) & (data['Lat'] < 50)]

# 将过滤后的数据点转换为 GeoDataFrame
points = gpd.GeoDataFrame(
    filtered_data,
    geometry=gpd.points_from_xy(filtered_data['Lon'], filtered_data['Lat']),
    crs=us_continental_map.crs
)

# 指定用于分组绘图的数据列名称
input_column = 'Fortan - gmt_offset'  # 请替换为实际的列名

# 指定分组值对应的颜色
color_mapping = {
    0: 'white',
    -1:'Red',
    1:'green',
    # 可以根据实际情况添加更多分组值和对应的颜色
}

# 根据用户指定的列分组
grouped = points.groupby(input_column)

# 创建绘图对象
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制美国本土地图
us_continental_map.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.5)

# 遍历每个分组并绘制数据点
for group_value, group in grouped:
    color = color_mapping.get(group_value, 'gray')  # 如果未指定颜色，使用灰色
    group.plot(ax=ax, color=color, markersize=6, alpha=0.7, label=f'{input_column} {group_value}')

# 添加图例
ax.legend()

# 添加标题
ax.set_title(f'US Continental Map with Monitoring Points by {input_column}', fontsize=16)

# 设置坐标轴标签
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图形
plt.show()
    
    
    
    
# import pandas as pd
# from shapely.geometry import Point
# from scipy.spatial import cKDTree

# # 读取包含监测点信息的 CSV 文件
# data = pd.read_csv('output/Region/MonitorsTimeRegion_Filter.csv')

# # 分离有 gmt_offset 和 epa_region 的点和没有这些信息的点
# known_points = data.dropna(subset=['gmt_offset', 'epa_region'])
# unknown_points = data[data['gmt_offset'].isna() | data['epa_region'].isna()]

# # 创建 GeoDataFrame
# known_gdf = pd.DataFrame({
#     'geometry': [Point(lon, lat) for lon, lat in zip(known_points['Lon'], known_points['Lat'])],
#     'gmt_offset': known_points['gmt_offset'],
#     'epa_region': known_points['epa_region']
# })
# unknown_gdf = pd.DataFrame({
#     'geometry': [Point(lon, lat) for lon, lat in zip(unknown_points['Lon'], unknown_points['Lat'])],
#     'index': unknown_points.index
# })

# # 构建 cKDTree 用于快速查找最近邻
# tree = cKDTree([(p.x, p.y) for p in known_gdf['geometry']])

# # 为每个未知点找到最近的已知点
# for _, row in unknown_gdf.iterrows():
#     point = row['geometry']
#     _, nearest_index = tree.query([point.x, point.y])
#     nearest_point = known_gdf.iloc[nearest_index]
#     # 直接使用值来填充，而不是访问 values 属性
#     data.loc[row['index'], 'gmt_offset'] = nearest_point['gmt_offset']
#     data.loc[row['index'], 'epa_region'] = nearest_point['epa_region']

# # 将处理后的数据保存到新的 CSV 文件
# output_file = 'output/Region/MonitorsTimeRegion_Filter_Add.csv'
# data.to_csv(output_file, index=False)
# print(f"处理后的数据已保存到 {output_file}")
    
    
    
# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('output/Region/MonitorsTimeRegion_Filter.csv')

# # 准备新的数据行，以字典形式提供，键为列名，值为对应的值
# new_row = {
#    'site_id': '80699991',  # 替换为实际的站点ID
#     'Lat': 40.2778,  # 替换为实际的纬度值
#     'Lon': -105.5453,  # 替换为实际的经度值
#     'gmt_offset': -7.0,  # 替换为实际的 GMT 偏移值
#     'epa_region': '8.0'  # 替换为实际的 EPA 区域
# }

# # 将新数据行添加到DataFrame中
# df = df.append(new_row, ignore_index=True)

# # 将修改后的数据写回到CSV文件
# df.to_csv('output/Region/MonitorsTimeRegion_Filter.csv', index=False) 
    
    