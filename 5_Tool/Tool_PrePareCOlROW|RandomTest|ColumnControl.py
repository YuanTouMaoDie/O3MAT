import pandas as pd
import os

# # 读取CSV文件，跳过前两行，从第三行开始加载数据，并指定列名
# df = pd.read_csv('/backupdata/data_EPA/EQUATES/EQUATES_data/Daily_EQUATES_CMAQv532_cb6r3_ae7_aq_STAGE_12US1_201607.csv', skiprows=2, names=[
#     'column', 'row', 'longitude', 'latitude', 'Lambert_X', 'LAMBERT_Y', 'date',
#     'O3_MDA8', 'O3_AVG', 'CO_AVG', 'NO_AVG', 'NO2_AVG', 'SO2_AVG', 'CH2O_AVG',
#     'PM10_AVG', 'PM25_AVG', 'PM25_SO4_AVG', 'PM25_NO3_AVG', 'PM25_NH4_AVG',
#     'PM25_OC_AVG', 'PM25_EC_AVG'
# ])

# # 重命名列
# df = df.rename(columns={
#     'column': 'COL',
#     'row': 'ROW',
#     'longitude': 'Lon',
#     'latitude': 'Lat'
# })

# # 提取所需的列并去重
# unique_df = df[['ROW', 'COL', 'Lon', 'Lat']].drop_duplicates()

# # 创建新的COL和ROW组合
# new_col, new_row = [], []
# for r in range(1, 300):
#     for c in range(1, 460):
#         new_row.append(r)
#         new_col.append(c)
# new_index_df = pd.DataFrame({'ROW': new_row, 'COL': new_col})

# # 使用merge方法合并数据，填充缺失值
# result = pd.merge(new_index_df, unique_df, on=['ROW', 'COL'], how='left')

# # 保存结果到新的CSV文件
# output_dir = 'output/Region'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# result.to_csv(os.path.join(output_dir, 'ROWCOLRegion.csv'), index=False)

#import pandas as pd

# 读取 CSV 文件
file_path = 'output/W126/2011_MonitorW126_Compare.csv'
try:
    df = pd.read_csv(file_path)

    # 添加新列
    df['SCUT - EPA: W126'] = df['SCUT_W126'] - df['EPA_W126']

    # 将修改后的数据保存回原文件
    df.to_csv(file_path, index=False)
    print(f"已成功在 {file_path} 中添加 'python - Fortan: gmt_offset' 列。")
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}。")
except KeyError:
    print("错误：数据中缺少 'python_gmt_offset' 或 'gmt_offset' 列。")
except Exception as e:
    print(f"发生未知错误：{e}")


# import pandas as pd
# import numpy as np

# # 生成1000行数据
# data = {
#    'site_id': [10030010] * 1000,
#     'Lat': np.random.uniform(25, 49, 1000),
#     'Lon': np.random.uniform(-125, -67, 1000),
#     'gmt_offset': [1] * 1000,
#     'epa_region': [1] * 1000,
#     'python_gmt_offset': [1] * 1000,
#     'python - Fortan: gmt_offset': [1] * 1000
# }

# df = pd.DataFrame(data)

# # 指定保存路径和文件名（这里以保存为当前目录下的test.csv为例）
# save_path = 'output/Region/MonitorsTimeRegion_Filter_ST_QA.csv'
# df.to_csv(save_path, index=False)

# import pandas as pd

# 读取CSV文件
df = pd.read_csv('output/Region/MonitorsTimeRegion_Filter_ST_QA_Compare.csv')

# # 确保列存在
# if 'Fortan_gmt_offset' in df.columns:
#     df['Fortan_gmt_offset'] = -df['Fortan_gmt_offset']
# else:
#     print("列 Fortan_gmt_offset 不存在于数据中。")

# 添加新列 Fortan - Python，为 Fortan_gmt_offset - python_gmt_offset
if 'Fortan_gmt_offset' in df.columns and 'gmt_offset' in df.columns:
    df['Fortan - gmt_offset'] = df['Fortan_gmt_offset'] - df['gmt_offset']
else:
    print("列 Fortan_gmt_offset 或 python_gmt_offset 不存在于数据中。")

# # 重命名列 python - Fortan: gmt_offset 为 Python - site_GMT_off
# if 'python - Fortan: gmt_offset' in df.columns:
#     df = df.rename(columns={'python - Fortan: gmt_offset': 'Python - site_GMT_off'})
# else:
#     print("列 python - Fortan: gmt_offset 不存在于数据中。")

# 将修改后的数据保存回原文件
df.to_csv('output/Region/MonitorsTimeRegion_Filter_ST_QA_Compare.csv', index=False)
# df.to_csv('new_file.csv', index=False)  # 保存为新文件，这里以new_file.csv为例



# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('output/Region/MonitorsTimeRegion_Filter_ST_QA_Compare.csv')

# # 要转换为整数的列名列表
# columns_to_convert = [
#     'gmt_offset', 'epa_region', 'python_gmt_offset',
#     'Python - site_GMT_off', 'Fortan_gmt_offset', 'Fortan - Python'
# ]

# for col in columns_to_convert:
#     if col in df.columns:
#         # 检查该列是否为数值类型
#         if pd.api.types.is_numeric_dtype(df[col]):
#             # 把浮点数转换为整数
#             df[col] = df[col].apply(lambda x: int(x) if x == int(x) else x)

# # 将修改后的数据保存回原文件
# df.to_csv('output/Region/MonitorsTimeRegion_Filter_ST_QA_Compare.csv', index=False)