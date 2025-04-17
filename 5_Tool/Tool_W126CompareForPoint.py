import pandas as pd
import re

# 读取第一个数据表
data1 = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2010_Monitor_W126.csv')

# 将 site_id 列转换为字符串类型
data1['site_id'] = data1['site_id'].astype(str)

# 去除 site_id 中的括号和引号
data1['site_id'] = data1['site_id'].str.strip("(')")

# 提取数字并去除开头的 0
data1['site_id'] = data1['site_id'].apply(lambda x: re.sub(r'[^0-9]', '', x).lstrip('0'))

# 读取第二个数据表
data2 = pd.read_csv('/backupdata/data_EPA/EQUATES/W126/SMAT_OZONE_W126_STD70_2000_2022.CSV')

# 筛选出 2011 年的数据
data2_2011 = data2[data2['YEAR'] == 2010]

# 提取数字并去除开头的 0
data2_2011['_ID'] = data2_2011['_ID'].astype(str).apply(lambda x: re.sub(r'[^0-9]', '', x).lstrip('0'))

# 合并两个数据表
merged_data = pd.merge(data1, data2_2011, left_on='site_id', right_on='_ID', how='inner')

# 汇总结果为一个表，包含站点、SCUT_W126 的 O3 值、EPA_W126 的 O3 值
result = merged_data[['site_id', 'O3_x', 'O3_y', 'LAT', 'LONG']]
result.columns = ['site_id', 'SCUT_W126', 'EPA_W126', 'Lat', 'Lon']

# 添加新列 SCUT_W126 - EPA_W126
result['SCUT_W126 - EPA_W126'] = result['SCUT_W126'] - result['EPA_W126']

# 去掉含有 NaN 的行
result = result.dropna()

# 剔除 SCUT_W126 或 EPA_W126 列中值为负数的行
result = result[(result['SCUT_W126'] >= 0) & (result['EPA_W126'] >= 0)]

# 输出结果
print(result)

# 保存结果到指定路径
output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2010_Monitor_W126_Compare.csv'
result.to_csv(output_path, index=False)
print(f"结果已保存到 {output_path}")