import pandas as pd
import numpy as np

# 定义文件路径
file_path_harvard = "/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronHarvard_ALL_2011_AtFtop.csv"
file_path_script = "/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronScript_ALL_2011_AtFtop.csv"

# 读取两个数据表
try:
    df_harvard = pd.read_csv(file_path_harvard)
    df_script = pd.read_csv(file_path_script)
except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")
    exit(1)

# 验证两个数据表的行数是否一致
if len(df_harvard) != len(df_script):
    print("两个数据表的行数不一致，请检查数据。")
    exit(1)

# 定义要处理的变量列
columns_to_process = ['evna_ozone']

# 验证列是否存在
for col in columns_to_process:
    if col not in df_harvard.columns or col not in df_script.columns:
        print(f"列 {col} 不存在于数据表中，请检查列名。")
        exit(1)

# 遍历每个变量列
for col in columns_to_process:
    # 找出 df_harvard 中该列值为 NaN 的索引
    nan_indices_harvard = df_harvard[col].isna()
    # 将 df_script 中对应索引位置的该列值也置为 NaN
    df_script.loc[nan_indices_harvard, col] = np.nan

# 保存处理后的数据表
output_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronScript_ALL_2011_AtFtop_FH.csv"
df_script.to_csv(output_path, index=False)

print(f"处理后的数据表已保存到 {output_path}")