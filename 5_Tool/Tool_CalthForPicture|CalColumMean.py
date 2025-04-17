# import pandas as pd
# import numpy as np
# import itertools

# # 数据文件路径
# data_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_Daily_Metrics.csv'

# # 读取数据文件
# df = pd.read_csv(data_path)

# # 指定要提取数据的 Period 值
# target_periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep']

# # 指定要提取的列
# target_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'harvard_ml', 'ds_ozone']
# # target_columns = ['SD']

# # 提取对应 Period 和列的数据
# extracted_data = df[df['Period'].isin(target_periods)][target_columns]

# # 将指定列的数据合并为一个一维数组
# combined_data = extracted_data.values.flatten()

# # 定义默认的最大值和最小值（可根据需要修改）
# default_max_value = None
# default_min_value = None

# # 计算 99.5 分位值
# vmax_conc = (
#     np.nanpercentile(combined_data, 99.5)
#     if default_max_value is None
#     else default_max_value
# )

# # 计算 0.5 分位值
# vmin_conc = (
#     np.nanpercentile(combined_data, 0.5)
#     if default_min_value is None
#     else default_min_value
# )

# # 计算整体数据的最大值和最小值
# overall_max = np.nanmax(combined_data)
# overall_min = np.nanmin(combined_data)

# print("整体数据的 99.5 分位值：", vmax_conc)
# print("整体数据的 0.5 分位值：", vmin_conc)
# print("整体数据的最大值：", overall_max)
# print("整体数据的最小值：", overall_min)

# # # 生成所有可能的列对
# # variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
# # comparisons = list(itertools.combinations(variables, 2))

# # # 存储所有列对的差值
# # all_differences = []

# # # 计算每对列的差值并存储
# # for col1, col2 in comparisons:
# #     diff = extracted_data[col1] - extracted_data[col2]
# #     all_differences.extend(diff)

# # # 将差值列表转换为 NumPy 数组
# # all_differences = np.array(all_differences)

# # # 计算所有差值的 99.5 分位值
# # diff_vmax_conc = (
# #     np.nanpercentile(all_differences, 99.5)
# #     if default_max_value is None
# #     else default_max_value
# # )

# # # 计算所有差值的 0.5 分位值
# # diff_vmin_conc = (
# #     np.nanpercentile(all_differences, 0.5)
# #     if default_min_value is None
# #     else default_min_value
# # )

# # # 计算所有差值数据的最大值和最小值
# # diff_max = np.nanmax(all_differences)
# # diff_min = np.nanmin(all_differences)

# # print("所有列对差值的 99.5 分位值：", diff_vmax_conc)
# # print("所有列对差值的 0.5 分位值：", diff_vmin_conc)
# # print("所有列对差值的最大值：", diff_max)
# # print("所有列对差值的最小值：", diff_min)

import pandas as pd


def calculate_stats(df, column_name, periods=None, period_column='Period'):
    """
    此函数用于计算指定列的统计信息，可根据 Period 列进行筛选。

    :param df: 输入的 DataFrame
    :param column_name: 要计算统计信息的列名
    :param periods: 可选，指定要筛选的 Period 值列表，默认为 None
    :param period_column: 可选，Period 列的列名，默认为 'Period'
    :return: 包含 Min、Max 和 Mean 的统计信息
    """
    if periods is not None and period_column in df.columns:
        df = df[df[period_column].isin(periods)]

    min_value = df[column_name].min()
    max_value = df[column_name].max()
    mean_value = df[column_name].mean()

    return round(min_value,2), round(max_value,2), round(mean_value,2)


# 读取 CSV 文件
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_W126_Compare.csv'
# file_path = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011_MissingDays.csv'
try:
    df = pd.read_csv(file_path)
    # 要计算统计信息的列名
    column_name = 'EPA_W126'

    # 计算统计信息，不指定 Period
    min_value, max_value, mean_value = calculate_stats(df, column_name)
    print(f"All Rows - Min= {min_value}, Max= {max_value}, Mean= {mean_value}")

    # # 定义要计算的多个 Period
    # periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr - Sep']

    # # 针对每个 Period 分别计算统计信息
    # for period in periods:
    #     min_value, max_value, mean_value = calculate_stats(df, column_name, periods=[period])
    #     print(f"Period {period} - Min= {min_value}, Max= {max_value}, Mean= {mean_value}")

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 未找到。")
except Exception as e:
    print(f"错误: 发生了未知错误: {e}")
    
    