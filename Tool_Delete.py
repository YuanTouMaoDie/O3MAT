import pandas as pd

file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_SixDataset_CONUSHarvard_metrics.csv'
try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 删除前缀带 Year 的列
    year_columns = [col for col in df.columns if col.startswith('Year')]
    df = df.drop(columns=year_columns)

    # 对于每行如果 vna_ozone 为 NaN，那么将 model 和 ds_ozone 也设为 NaN
    df.loc[df['vna_ozone'].isna(), ['model', 'ds_ozone']] = float('nan')

    # 将处理后的数据保存回原文件
    df.to_csv(file_path, index=False)
    print("处理完成。")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到。")
    