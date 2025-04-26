import pandas as pd

# 定义文件路径
input_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/rn_filtered_data.csv"
output_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/new_data.csv"

# 读取 CSV 文件
df = pd.read_csv(input_path)

# 将 localtime 列转换为 datetime 类型
df['localtime'] = pd.to_datetime(df['localtime'])

# 统计最大的 r_n 和对应的时间段、id、行号
max_r_n = df['r_n'].max()
max_r_n_rows = df[df['r_n'] == max_r_n]
max_r_n_time = max_r_n_rows['localtime'].tolist()
max_r_n_id = max_r_n_rows['site_id'].tolist()
max_r_n_row_numbers = max_r_n_rows.index.tolist()

print(f"整个数据集中最大的 r_n 值为 {max_r_n}，对应的时间段为 {max_r_n_time}，对应的 site_id 为 {max_r_n_id}，对应的原数据表行号为 {max_r_n_row_numbers}")

# 未剔除数据时的最大 r_n
max_r_n_without_filter = df['r_n'].max()

# 筛选出早 8 点到晚 8 点的数据
mask = (df['localtime'].dt.hour >= 8) & (df['localtime'].dt.hour <= 20)
df_filtered = df[mask]

# 剔除数据后的最大 r_n
max_r_n_after_filter = df_filtered['r_n'].max()

# 被剔除部分数据
df_excluded = df[~mask]

# 被剔除部分数据的最大 r_n
max_r_n_excluded = df_excluded['r_n'].max()

# 按 site_id 分组并计算平均值
result = df_filtered.groupby('site_id').mean()

# 输出结果到新的 CSV 文件
result.to_csv(output_path)

print(f"未剔除数据时的最大 r_n 值: {max_r_n_without_filter}")
print(f"剔除早 8 点到晚 8 点以外数据后的最大 r_n 值: {max_r_n_after_filter}")
print(f"被剔除部分数据的最大 r_n 值: {max_r_n_excluded}")
