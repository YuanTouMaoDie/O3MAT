import pandas as pd

# 读取 CSV 文件
try:
    df = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_W126.csv')
    # 添加 Flag 列
    df['Flag'] = df['O3'].apply(lambda x: 2 if pd.notna(x) else 1)
    # 保存修改后的数据到新的 CSV 文件
    df.to_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_W126.csv', index=False)
    print("Flag 列已成功添加，结果保存到 2011_Monitor_W126_with_flag.csv 文件中。")
except FileNotFoundError:
    print("错误：未找到 2011_Monitor_W126.csv 文件，请检查文件路径。")
except KeyError:
    print("错误：数据中没有 'O3' 列，请检查数据。")
except Exception as e:
    print(f"发生未知错误：{e}")
    