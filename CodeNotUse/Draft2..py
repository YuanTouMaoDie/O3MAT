import os
import pandas as pd
from datetime import datetime

def merge_data_file(model_file, year, start_date=None, end_date=None):
    # 读取模型数据，跳过第一行并将第二行作为列名
    df_model = pd.read_csv(model_file, header=None, skiprows=1)

    # 设置列名（根据你的文件结构，这里设置适当的列名）
    # df_model.columns = ['_id', 'gridcell_lat', 'gridcell_long', 'day', 'O3_Model', 'O3_VNA', 'O3_eVNA']
    df_model.columns = ['_id','O3_aVNA','day']
    # 打印df_model的列名，检查是否有需要的列
    print("Columns in df_model:", df_model.columns)

    # 对原始数据按照 day 和 _id 排序，确保数据顺序正确
    print("Sorting original data by day and _id...")
    df_model = df_model.sort_values(by=['day', '_id']).reset_index(drop=True)

    # 提取包含日期数据的 'day' 列
    day_data = df_model['day'].dropna().reset_index(drop=True)
    print(f"Found {len(day_data.unique())} unique days in the data.")

    # 创建一个空的DataFrame来存储最终的合并结果
    merged_df = pd.DataFrame()

    # 如果提供了日期范围，将其转换为日期格式
    if start_date:
        start_date = pd.to_datetime(start_date)
        print(f"Start date: {start_date}")
    if end_date:
        end_date = pd.to_datetime(end_date)
        print(f"End date: {end_date}")

    # 每个 day 列表对应 459行 * 299列 (需要交换)
    rows_per_col = 459  # 每列有459行 (ROW的数量)
    cols_per_period = 299  # 每个周期最多299列 (COL的数量)

    for day in day_data.unique():  # 逐个日期处理
        # 确保 day 是字符串类型
        day_str = str(day).zfill(4)  # 转换为字符串并确保至少有4位（例如 '0101'，'1230'）

        # 构造日期字符串，比如 '0101' -> '2011-01-01'
        date_str = f"{year}-{day_str[:2]}-{day_str[2:]}"  # '2011-01-01'

        # 转换为日期格式
        timestamp = pd.to_datetime(date_str)
        print(f"Processing day: {timestamp}")

        # 如果设置了日期范围，跳过不在范围内的日期
        if (start_date and timestamp < start_date) or (end_date and timestamp > end_date):
            print(f"Skipping {timestamp} (outside the specified date range)")
            continue  # 跳过不在日期范围内的日期

        # 获取VNA和EVNA数据
        vna_data = df_model.loc[df_model['day'] == day, 'O3_aVNA'].dropna().reset_index(drop=True)
        print(len(vna_data))
        # evna_data = df_model.loc[df_model['day'] == day, 'O3_eVNA'].dropna().reset_index(drop=True)

        # 检查数据长度是否匹配
        if len(vna_data) != rows_per_col * cols_per_period:
            print(f"Warning: Data length for {day} does not match expected size. Skipping this day.")
            continue  # 跳过这个不匹配的数据

        print(f"Data for {day} is valid. Length: {len(vna_data)}")

        # 创建包含ROW和COL的列（交换ROW和COL的计算方式）
        rows = [(i % rows_per_col) + 1 for i in range(len(vna_data))]  # ROW从1到459
        cols = [(i // rows_per_col) + 1 for i in range(len(vna_data))]  # COL从1到299

        # 构建临时DataFrame
        temp_df = pd.DataFrame({
            'ROW': rows,
            'COL': cols,
            'avna_ozone': vna_data,
            'Timestamp': [timestamp] * len(vna_data)
        })

        # 合并当前数据
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # 计算每个Timestamp对应的数据行数
    row_data = []  # 用来存储计算出的ROW和COL
    for timestamp, group in merged_df.groupby('Timestamp'):
        length = len(group)  # 获取每个Timestamp的行数

        # 为每个 Timestamp 单独计算 ROW 和 COL
        rows = [(i // cols_per_period) + 1 for i in range(length)]  # ROW从1到299
        cols_list = [(i % cols_per_period) + 1 for i in range(length)]  # COL从1到459

        # 将 ROW 和 COL 赋值到对应的数据行
        group['ROW'] = rows
        group['COL'] = cols_list

        # 将这个 group 添加到最终数据框中
        row_data.append(group)

    # 合并所有的group
    merged_df = pd.concat(row_data, ignore_index=True)

    # 将COL和ROW交换位置（如果需要）
    merged_df['ROW'], merged_df['COL'] = merged_df['COL'], merged_df['ROW']

    # 重新按照ROW和COL排序
    merged_df = merged_df.sort_values(by=['Timestamp', 'ROW', 'COL']).reset_index(drop=True)

    # 现在可以返回已排序的结果，不需要重复的列放置操作
    print(f"Processed data shape: {merged_df.shape}")

    return merged_df

# 输入文件路径
# data_fusion_dir = "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_daily_data_fusion/"
# model_file = os.path.join(data_fusion_dir, "2011_dailyIn_dailyOut_O3_Ozone_Data.csv")
data_fusion_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/"
model_file = os.path.join(data_fusion_dir, "DFT_aVNA_20110101.csv")

# 设置年份（例如 2011）
year = 2011

# 设置日期范围（例如 '2011-01-01' 到 '2011-01-02'）
start_date = '2011-01-01'
end_date = '2011-01-01'

# 调用函数并合并数据
print("Starting to merge data...")
merged_df = merge_data_file(model_file, year, start_date=start_date, end_date=end_date)

# 保存合并后的数据
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DFT_aVNA_20110101_daily.csv'
merged_df.to_csv(output_file, index=False)
print(f"数据已成功合并并保存到 {output_file}")
