import os
import pandas as pd

def merge_data_files(file_list, year_prefix):
    # 创建一个空的DataFrame来存储最终的合并结果
    merged_df = pd.DataFrame()

    # 文件名到Period的映射
    period_mapping = {
        'quarter_1': 'DJF',
        'quarter_2': 'MAM',
        'quarter_3': 'JJA',
        'quarter_4': 'SON',
        'Annual': 'Annual',
        'summer': 'Apr-Sep'
    }

    for file_path in file_list:
        # 从文件名中提取Period信息
        file_name = os.path.basename(file_path)
        period_key = None
        for key in period_mapping:
            if key in file_name:
                period_key = key
                break

        if period_key is None:
            continue  # 如果文件名中没有匹配的Period信息，跳过该文件

        period_name = f"{year_prefix}_{period_mapping[period_key]}"

        # 读取文件，跳过第一行
        df = pd.read_csv(file_path, skiprows=1)

        # 确保文件中有需要的列
        if not all(col in df.columns for col in ['O3_Model', 'O3_VNA', 'O3_eVNA', '_id']):
            continue  # 如果缺少必要的列，跳过该文件

        # 提取需要的列
        temp_df = df[['_id', 'O3_Model', 'O3_VNA', 'O3_eVNA']].copy()
        temp_df.rename(columns={'O3_Model': 'model', 'O3_VNA': 'vna_ozone', 'O3_eVNA': 'evna_ozone'}, inplace=True)
        temp_df['Period'] = period_name

        # 合并当前文件的数据
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # 按照Period和_id排序
    merged_df = merged_df.sort_values(by=['Period', '_id']).reset_index(drop=True)

    # 在最终合并的数据表上计算ROW和COL
    length = len(merged_df)  # 获取最终数据表的长度
    rows_per_col = 299  # ROW从1到299
    cols = (length - 1) // rows_per_col + 1  # COL最大值（459）

    # 为整个数据表生成ROW和COL
    rows = []
    cols_list = []
    for i in range(length):
        row = (i // rows_per_col) + 1  # COL从1开始，最大到459
        col = (i % rows_per_col) + 1  # ROW从1开始，最大到299
        cols_list.append(col)
        rows.append(row)

    # 将ROW和COL列添加到merged_df中
    merged_df['COL'] = cols_list
    merged_df['ROW'] = rows

    # 交换ROW和COL的数值
    merged_df['ROW'], merged_df['COL'] = merged_df['COL'], merged_df['ROW']

    # 重新按照Period、ROW和COL排序
    merged_df = merged_df.sort_values(by=['Period', 'ROW', 'COL']).reset_index(drop=True)

    # 将_id、ROW和COL列放到前面
    cols = ['_id', 'ROW', 'COL', 'model', 'vna_ozone', 'evna_ozone', 'Period']
    merged_df = merged_df[cols]

    return merged_df

# 输入文件路径
data_fusion_dir = "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion"
file_list = [
    os.path.join(data_fusion_dir, "2011_dailyIn_AnnualOut_O3_Ozone_Data.csv"),
    os.path.join(data_fusion_dir, "test_new_quarter_1_O3_Ozone_Data.csv"),
    os.path.join(data_fusion_dir, "test_new_quarter_2_O3_Ozone_Data.csv"),
    os.path.join(data_fusion_dir, "test_new_quarter_3_O3_Ozone_Data.csv"),
    os.path.join(data_fusion_dir, "test_new_quarter_4_O3_Ozone_Data.csv"),
    os.path.join(data_fusion_dir, "summer_copy.csv")
]

# 自定义年份前缀
year_prefix = "2011"  # 可以修改为其他年份，比如 "2012" 等

# 调用函数并合并数据
merged_df = merge_data_files(file_list, year_prefix)

# 保存合并后的数据
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_seasonal_data_fusion_DFT_AI.csv'
merged_df.to_csv(output_file, index=False)
print(f"数据已成功合并并保存到 {output_file}")