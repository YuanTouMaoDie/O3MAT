# import vaex

# # 指定要读取的列
# usecols = ['ROW', 'COL', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'model', 'Timestamp']

# # 读取第一个 CSV 文件，仅读取指定列
# df1 = vaex.open('//DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly_True_Duan.csv',
#                 columns=usecols)

# # 将 Timestamp 列转换为 datetime 类型
# df1['Timestamp'] = df1['Timestamp'].astype('datetime64[ns]')

# # 删除 Timestamp 大于等于 2011-08-10 20:00 的数据行
# df1 = df1[df1['Timestamp'] < '2011-08-10 20:00']

# # 读取第二个 CSV 文件，仅读取指定列
# df2 = vaex.open('/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Hourly_True_BU.csv',
#                 columns=usecols)

# # 上下拼接两个 DataFrame
# combined_df = vaex.concat([df1, df2])

# # 保存拼接后的 DataFrame 到新的 CSV 文件
# combined_df.export_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/combined_data_vaex.csv')
# import pandas as pd


#Find COL MAX
# import pandas as pd
# import pandas as pd


# def get_top_and_bottom_three_rows(df, column_name):
#     # 筛选 vna_ozone 列有数据的行
#     valid_df = df[df['vna_ozone'].notna()]
#     # 按照指定列降序排序
#     sorted_desc = valid_df.sort_values(by=column_name, ascending=False)
#     # 获取最大、第二大、第三大的行
#     top_three = sorted_desc.head(3)
#     print(top_three)
#     # 按照指定列升序排序
#     sorted_asc = valid_df.sort_values(by=column_name, ascending=True)
#     # 获取最小、第二小、第三小的行
#     bottom_three = sorted_asc.head(3)
#     return pd.concat([top_three, bottom_three])


# def read_data(file_path):
#     try:
#         if file_path.endswith('.csv'):
#             return pd.read_csv(file_path)
#         elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#             return pd.read_excel(file_path)
#         elif file_path.endswith('.json'):
#             return pd.read_json(file_path)
#         else:
#             print("不支持的文件格式，请提供 CSV、Excel 或 JSON 文件。")
#             return None
#     except FileNotFoundError:
#         print(f"错误: 文件 {file_path} 未找到。")
#         return None
#     except Exception as e:
#         print(f"错误: 读取文件时出现问题: {e}")
#         return None


# # 读取文件
# file_path = 'output/2011_Data_WithoutCV/2011_SixDataset_Hourly_Test.csv'  # 替换为实际的文件路径
# df = read_data(file_path)

# if df is not None:
#     # 调用函数
#     result = get_top_and_bottom_three_rows(df, 'COL')
#     print(result)


#Delete Column

# import pandas as pd

# def delete_columns_and_override(file_path, columns_to_delete):
#     try:
#         # 读取 CSV 文件
#         df = pd.read_csv(file_path)

#         # 删除指定列
#         df = df.drop(columns=columns_to_delete, errors='ignore')

#         # 覆盖原文件
#         df.to_csv(file_path, index=False)
#         print(f"已成功删除指定列并覆盖原文件: {file_path}")
#     except FileNotFoundError:
#         print(f"错误: 文件 {file_path} 未找到。")
#     except Exception as e:
#         print(f"错误: 发生了一个未知错误: {e}")

# # 示例调用
# file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Data_WithoutCV/2002_Data_WithoutCV_Metrics.csv'
# columns_to_delete = ['Year_x','Year_y','harvard_ml', 'ds_ozone']
# delete_columns_and_override(file_path, columns_to_delete)

import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/ROWCOLRegion_Tz_(CONUS+Ocean)_ST.csv')
df2 = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/ROWCOLRegion_Tz_CONUS_ST.csv')

# 合并两个数据帧，基于ROW和COL列
merged_df = pd.merge(df1, df2, on=['ROW', 'COL'], suffixes=('_df1', '_df2'))

# 计算差异（Diff）
merged_df['Diff'] = merged_df['gmt_offset_df1'] - merged_df['gmt_offset_df2']

# 选择需要的列并重新命名
result_df = merged_df[['ROW', 'COL', 'Lat_df1', 'Lon_df1', 'Diff']].rename(columns={
    'Lat_df1': 'Lat',
    'Lon_df1': 'Lon'
})

# 保存结果到指定路径
result_df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/OceanQA.csv', index=False)