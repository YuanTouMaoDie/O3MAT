import pandas as pd
import os

# 读取CSV文件，跳过前两行，从第三行开始加载数据，并指定列名
df = pd.read_csv('/backupdata/data_EPA/EQUATES/EQUATES_data/Daily_EQUATES_CMAQv532_cb6r3_ae7_aq_STAGE_12US1_201607.csv', skiprows=2, names=[
    'column', 'row', 'longitude', 'latitude', 'Lambert_X', 'LAMBERT_Y', 'date',
    'O3_MDA8', 'O3_AVG', 'CO_AVG', 'NO_AVG', 'NO2_AVG', 'SO2_AVG', 'CH2O_AVG',
    'PM10_AVG', 'PM25_AVG', 'PM25_SO4_AVG', 'PM25_NO3_AVG', 'PM25_NH4_AVG',
    'PM25_OC_AVG', 'PM25_EC_AVG'
])

# 重命名列
df = df.rename(columns={
    'column': 'COL',
    'row': 'ROW',
    'longitude': 'Lon',
    'latitude': 'Lat'
})

# 提取所需的列并去重
unique_df = df[['ROW', 'COL', 'Lon', 'Lat']].drop_duplicates()

# 创建新的COL和ROW组合
new_col, new_row = [], []
for r in range(1, 300):
    for c in range(1, 460):
        new_row.append(r)
        new_col.append(c)
new_index_df = pd.DataFrame({'ROW': new_row, 'COL': new_col})

# 使用merge方法合并数据，填充缺失值
result = pd.merge(new_index_df, unique_df, on=['ROW', 'COL'], how='left')

# 保存结果到新的CSV文件
output_dir = 'output/Region'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
result.to_csv(os.path.join(output_dir, 'ROWCOLRegion.csv'), index=False)