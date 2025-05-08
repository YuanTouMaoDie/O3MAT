import pandas as pd

# 定义文件路径
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/2008_Data_WithoutCV_Metrics.csv'

try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 获取要删除的列名
    columns_to_delete = ['ds_ozone', 'harvard_ml']
    columns_to_delete = ['ds_ozone','harvard_ml']
    columns_to_delete.extend([col for col in df.columns if col.startswith('Year')])

    # 删除指定列
    df = df.drop(columns=columns_to_delete)

    # 保存修改后的数据到原文件
    df.to_csv(file_path, index=False)
    print(f"已成功删除 {columns_to_delete} 列，并保存修改后的数据到 {file_path}")

except FileNotFoundError:
    print(f"错误：未找到指定的 CSV 文件 {file_path}，请检查文件路径是否正确。")
except Exception as e:
    print(f"处理文件时发生错误：{e}")

# import pandas as pd

# # 定义文件路径
# file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/2011_Data_WithoutCV_Metrics.csv'

# try:
#     # 读取 CSV 文件
#     df = pd.read_csv(file_path)

#     # 定义要重命名的列及其新名称
#     column_rename_mapping = {
#         'harvard_ml_x': 'harvard_ml',
#         'ds_ozone_x': 'ds_ozone'
#         # 你可以继续添加需要重命名的列
#     }

#     # 重命名列
#     df = df.rename(columns=column_rename_mapping)

#     # 保存修改后的数据到原文件
#     df.to_csv(file_path, index=False)
#     print(f"已成功重命名 {list(column_rename_mapping.keys())} 列为 {list(column_rename_mapping.values())}，并保存修改后的数据到 {file_path}")

# except FileNotFoundError:
#     print(f"错误：未找到指定的 CSV 文件 {file_path}，请检查文件路径是否正确。")
# except Exception as e:
#     print(f"处理文件时发生错误：{e}")