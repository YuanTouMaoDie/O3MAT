import pandas as pd
import re

# 读取第二个数据表
data2 = pd.read_csv('/backupdata/data_EPA/EQUATES/W126/SMAT_OZONE_W126_STD70_2000_2022.CSV')

# 筛选出 2011 年的数据
data2_2011 = data2[data2['YEAR'] == 2011]

# 提取数字并去除开头的 0
data2_2011['_ID'] = data2_2011['_ID'].astype(str).apply(lambda x: re.sub(r'[^0-9]', '', x).lstrip('0'))

# 从 data2_2011 中提取需要的列并进行重命名
result = data2_2011[['_ID', 'LAT', 'LONG', 'O3']]
result = result.rename(columns={'LAT': 'Lat', 'LONG': 'Lon', '_ID': 'Site'})

# 去掉含有 NaN 的行
result = result.dropna()

# 统计总的站点数
total_sites = len(result)

# 统计 O3 列中值小于 0 的站点数量
negative_O3_sites = len(result[result['O3'] < 0])

# 输出统计结果
print(f"总的站点数: {total_sites}")
print(f"O3 值小于 0 的站点数: {negative_O3_sites}")

# 剔除 O3 列中值为负数的行
result = result[result['O3'] >= 0]

# 输出结果
print(result)

# 保存结果到指定路径
output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_MonitorW126_SMAT_WithNaN.csv'
result.to_csv(output_path, index=False)
print(f"结果已保存到 {output_path}")
    