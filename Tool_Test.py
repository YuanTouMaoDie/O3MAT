import pandas as pd

# 目标文件路径
existing_df_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/BarronScript_ALL_2011_FtAIndex_CONUS.csv'

# 读取 CSV 文件
df = pd.read_csv(existing_df_path)

# 修改列名

# 删除所有以 "Year" 开头的列
df.drop(columns=('harvard_ml.1'), inplace=True)

# 直接覆盖原文件
df.to_csv(existing_df_path, index=False)

