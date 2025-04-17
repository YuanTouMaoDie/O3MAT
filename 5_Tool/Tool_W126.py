import pandas as pd

# 读取 CSV 文件
file_path = '/backupdata/data_EPA/EQUATES/W126/W126_CMAQ_2002_2019_March_October.csv'
df = pd.read_csv(file_path)

# 提取需要的列
columns_to_extract = ['X_id', 'lat', 'long', 'W126_2010_CMAQ']
extracted_df = df[columns_to_extract].copy()

# 定义函数来转换 X_id 为 ROW 和 COL
def convert_x_id(x_id):
    x_id_str = str(x_id)
    row = int(x_id_str[-3:])
    col = int(x_id_str[:-3]) if len(x_id_str) > 3 else 0
    return row, col

# 应用转换函数
extracted_df[['ROW', 'COL']] = extracted_df['X_id'].apply(lambda x: pd.Series(convert_x_id(x)))

# 删除原始的 X_id 列
extracted_df.drop('X_id', axis=1, inplace=True)

# 调整列顺序
new_column_order = ['ROW', 'COL', 'lat', 'long', 'W126_2010_CMAQ']
extracted_df = extracted_df[new_column_order]

# 按 ROW 和 COL 排序
extracted_df = extracted_df.sort_values(by=['ROW', 'COL'])

# 将 W126_2011_CMAQ 列重命名为 VNA
extracted_df = extracted_df.rename(columns={'W126_2010_CMAQ': 'model'})

# 新加入一列 Period，值全为 W126
extracted_df['Period'] = 'W126'

# 调整最终的列顺序（如果需要）
final_column_order = ['ROW', 'COL', 'lat', 'long', 'model', 'Period']
extracted_df = extracted_df[final_column_order]

# 如果需要保存为新的 CSV 文件
extracted_df.to_csv('output/W126/W126_CMAQ_2010_March_October.csv', index=False)