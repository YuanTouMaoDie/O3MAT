import os
import pandas as pd

def add_row_col_columns(model_file):
    # 读取模型数据，第一行作为列名
    df_model = pd.read_csv(model_file)

    # 假设数据的行数为 rows，这里先获取数据的行数
    rows = len(df_model)
    # 这里假设你希望按一个固定的列数进行 COL 的循环，比如 10 列，你可以根据实际情况修改
    num_cols = 268

    # 生成 ROW 和 COL 列的数据
    row_list = []
    col_list = []
    for i in range(rows):
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1
        row_list.append(row)
        col_list.append(col)

    # 添加 ROW 和 COL 列到数据框
    df_model['ROW'] = row_list
    df_model['COL'] = col_list

    return df_model

# 输入文件路径
data_fusion_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/OTHER"
model_file = os.path.join(data_fusion_dir, "Input/ModelO3andPM25.csv")

# 调用函数添加列
print("Starting to add ROW and COL columns...")
new_df = add_row_col_columns(model_file)

# 保存处理后的数据
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/OTHER/Model.csv'
new_df.to_csv(output_file, index=False)
print(f"数据已成功添加 ROW 和 COL 列并保存到 {output_file}")
    