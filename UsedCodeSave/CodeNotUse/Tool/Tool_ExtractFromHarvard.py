import pandas as pd
import os

def process_non_specified_region(file_path_1, file_path_2):
    try:
        # 从 CSV 文件读取数据
        table1 = pd.read_csv(file_path_1)
        table2 = pd.read_csv(file_path_2)

        # 筛选出 table2 中 Is 列为空的数据
        invalid_region = table2[table2['Is'].isna()]

        # 提取无效的 ROW 和 COL 组合
        invalid_coords = invalid_region[['ROW', 'COL']]

        # 创建一个布尔索引，标记 table1 中属于无效区域的行
        invalid_mask = table1[['ROW', 'COL']].apply(tuple, axis=1).isin(invalid_coords.apply(tuple, axis=1))

        # 获取除 ROW、COL、Timestamp 之外的列名
        non_key_columns = [col for col in table1.columns if col not in ['ROW', 'COL', 'Timestamp']]

        # 将无效区域对应行的非关键列的值设为 NaN
        table1.loc[invalid_mask, non_key_columns] = pd.NA

        return table1
    except FileNotFoundError:
        print("错误：未找到指定的 CSV 文件。")
    except Exception as e:
        print(f"发生未知错误：{e}")


# 有 Is 列数据表路径
file_path_2 = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv'
# 没有 Is 列数据表路径
file_path_1 = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DFT_Test/1_Trans.csv'

# 处理数据表
result = process_non_specified_region(file_path_1, file_path_2)

if result is not None:
    # 获取原文件名和扩展名
    file_name, file_ext = os.path.splitext(file_path_1)
    # 构建新文件名
    new_file_name = f"{file_name}CONUS{file_ext}"
    # 将结果保存为 CSV 文件
    result.to_csv(new_file_name, index=False)
    print(f"处理后的数据已保存到 {new_file_name}")
    