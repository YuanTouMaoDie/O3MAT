import pandas as pd

def check_duplicate_rows(data):
    # 按 Timestamp、Fold 和 Method 分组
    grouped = data.groupby(['Timestamp', 'Fold', 'Method'])

    # 遍历每个组
    for group_name, group_data in grouped:
        # 检查 ROW 和 COL 列是否有重复的行
        duplicate_mask = group_data.duplicated(subset=['ROW', 'COL'], keep=False)
        duplicate_rows = group_data[duplicate_mask]

        # 如果存在重复行
        if not duplicate_rows.empty:
            print(f"在 Timestamp: {group_name[0]}, Fold: {group_name[1]}, Method: {group_name[2]} 下发现重复行:")
            # 按 ROW 和 COL 再次分组，以便输出每一组重复的两行
            duplicate_groups = duplicate_rows.groupby(['ROW', 'COL'])
            for _, dup_group in duplicate_groups:
                print(dup_group)
        else:
            print(f"在 Timestamp: {group_name[0]}, Fold: {group_name[1]}, Method: {group_name[2]} 下未发现重复行。")

# 请替换为你的 CSV 文件路径
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/CV_Data/BarronScript_2011_ALL_FtA0101_CV_results.csv'

try:
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    # 筛选 Method 列为 evna_ozone 的数据
    filtered_df = df[df['Method'] == 'evna_ozone']
    if not filtered_df.empty:
        # 调用函数进行验证
        check_duplicate_rows(filtered_df)
    else:
        print("数据中没有 Method 列为 evna_ozone 的记录。")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查文件路径。")
except KeyError:
    print("数据中缺少必要的列（Timestamp、Fold、Method、ROW、COL），请检查数据。")