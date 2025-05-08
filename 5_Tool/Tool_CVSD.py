import pandas as pd

# 数据文件路径
data_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_W126_1.csv"

# 读取数据
df = pd.read_csv(data_path)

# 假设参与计算的列，你需要根据实际情况修改
# columns_to_calculate = ['model','vna_ozone', 'evna_ozone', 'avna_ozone','ds_ozone','harvard_ml']
#colums_W126
columns_to_calculate = ['model','vna_ozone', 'evna_ozone', 'avna_ozone','ds_ozone','harvard_ml']

# 用于存储结果的数据列表
results = []

# 按 Period 分组进行计算
for period, group in df.groupby('Period'):
    # 按 ROW 和 COL 分组计算标准偏差和变异系数
    for (row, col), sub_group in group.groupby(['ROW', 'COL']):
        # 提取参与计算的列数据
        calc_data = sub_group[columns_to_calculate]
        # 检查是否全为 NaN
        if calc_data.isna().all().all():
            std_dev = float('nan')
            mean_value = float('nan')
        else:
            # 计算标准偏差
            std_dev = calc_data.std(axis=1).mean()
            # 计算均值
            mean_value = calc_data.mean(axis=1).mean()

        # 计算变异系数
        cv = (std_dev / mean_value) * 100 if pd.notna(mean_value) and mean_value != 0 else float('nan')

        results.append([row, col, std_dev, cv, period])

# 创建结果数据框
result_df = pd.DataFrame(results, columns=['ROW', 'COL', 'SD', 'CV', 'Period'])

# 输出结果数据框
print(result_df)

# 如果需要保存结果到文件
result_df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_W126_CVSD.csv', index=False)
    