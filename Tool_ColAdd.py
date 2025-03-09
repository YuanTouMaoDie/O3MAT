import pandas as pd

# 定义 CSV 文件的路径
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronScript_ALL_2011_FtAIndex_top10fs.csv'

# 读取 CSV 文件
df = pd.read_csv(file_path)

# 添加新列 'Period'，并将其值设置为 'top-10'
df['Period'] = 'top-10'

# 将修改后的数据保存回原文件
df.to_csv(file_path, index=False)

print("已成功添加 'Period' 列，值均为 'top-10'。")