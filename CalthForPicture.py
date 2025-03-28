import pandas as pd
import numpy as np
import itertools

# 数据文件路径
data_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_CVSD.csv'

# 读取数据文件
df = pd.read_csv(data_path)

# 指定要提取数据的 Period 值
target_periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep']

# 指定要提取的列
# target_columns = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'harvard_ml', 'ds_ozone']
target_columns = ['SD']

# 提取对应 Period 和列的数据
extracted_data = df[df['Period'].isin(target_periods)][target_columns]

# 将指定列的数据合并为一个一维数组
combined_data = extracted_data.values.flatten()

# 定义默认的最大值和最小值（可根据需要修改）
default_max_value = None
default_min_value = None

# 计算 99.5 分位值
vmax_conc = (
    np.nanpercentile(combined_data, 99.5)
    if default_max_value is None
    else default_max_value
)

# 计算 0.5 分位值
vmin_conc = (
    np.nanpercentile(combined_data, 0.5)
    if default_min_value is None
    else default_min_value
)

# 计算整体数据的最大值和最小值
overall_max = np.nanmax(combined_data)
overall_min = np.nanmin(combined_data)

print("整体数据的 99.5 分位值：", vmax_conc)
print("整体数据的 0.5 分位值：", vmin_conc)
print("整体数据的最大值：", overall_max)
print("整体数据的最小值：", overall_min)

# # 生成所有可能的列对
# variables = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']
# comparisons = list(itertools.combinations(variables, 2))

# # 存储所有列对的差值
# all_differences = []

# # 计算每对列的差值并存储
# for col1, col2 in comparisons:
#     diff = extracted_data[col1] - extracted_data[col2]
#     all_differences.extend(diff)

# # 将差值列表转换为 NumPy 数组
# all_differences = np.array(all_differences)

# # 计算所有差值的 99.5 分位值
# diff_vmax_conc = (
#     np.nanpercentile(all_differences, 99.5)
#     if default_max_value is None
#     else default_max_value
# )

# # 计算所有差值的 0.5 分位值
# diff_vmin_conc = (
#     np.nanpercentile(all_differences, 0.5)
#     if default_min_value is None
#     else default_min_value
# )

# # 计算所有差值数据的最大值和最小值
# diff_max = np.nanmax(all_differences)
# diff_min = np.nanmin(all_differences)

# print("所有列对差值的 99.5 分位值：", diff_vmax_conc)
# print("所有列对差值的 0.5 分位值：", diff_vmin_conc)
# print("所有列对差值的最大值：", diff_max)
# print("所有列对差值的最小值：", diff_min)