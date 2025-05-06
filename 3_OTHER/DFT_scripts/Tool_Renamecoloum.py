import pandas as pd

# 读取CSV文件
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/2017_PM25_DFTAtFAnnual_PythonFormat_CONUS.csv'
df = pd.read_csv(file_path)

# 定义列名映射
column_mapping = {
    'vna_ozone': 'vna_pm25',
    'evna_ozone': 'evna_pm25',
    'avna_ozone': 'avna_pm25'
}

# 重命名列名
df = df.rename(columns=column_mapping)

# 保存修改后的数据到新的CSV文件
output_file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/2017_PM25_DFTAtFAnnual_PythonFormat_CONUS.csv'
df.to_csv(output_file_path, index=False)

print(f"列名已成功重命名，修改后的数据保存到 {output_file_path}")    