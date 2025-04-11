import os
import pandas as pd

# 定义目录路径
directory_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/"

# 定义需要合并的文件名
file_names = [
    "2011_FtA_Python_filteredForMap_ROWCOL_JSON.csv",
    "2011_FtA_Python_filteredForMap_ROWCOL_JSON_Special.csv",
]

# 构建完整的文件路径
file_paths = [os.path.join(directory_path, file) for file in file_names]

# 检查文件是否存在
missing_files = [file for file in file_paths if not os.path.exists(file)]
if missing_files:
    print("The following files are missing:")
    for file in missing_files:
        print(file)
else:
    # 读取所有文件并存储到列表中
    dataframes = [pd.read_csv(file) for file in file_paths]

    # 合并所有 DataFrame
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

    # 输出合并后的结果
    print("Combined DataFrame:")
    print(combined_df)

    # 保存合并后的数据到新文件
    output_file = os.path.join(directory_path, "2011_FtA_Python_filteredForMap_ROWCOL_JSON_ALL.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")