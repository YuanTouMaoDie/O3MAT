import pandas as pd


def process_file(input_file_path, output_file_path):
    # 读取文件，跳过第一行
    df = pd.read_csv(input_file_path)
    print("文件实际列名:", df.columns)  # 打印实际列名

    # 所需列列表
    # required_columns = ['pm25_Model', 'pm25_VNA', 'pm25_eVNA', 'pm25_aVNA', '_id']
    required_columns = ['pm25_Model_mAverage', 'pm25_VNA_mAverage', 'pm25_eVNA_mAverage', 'pm25_aVNA_mAverage', '_id']
    print("需要的列名:", required_columns)  # 打印需要的列名

    # 过滤出文件中实际存在的所需列
    existing_columns = [col for col in required_columns if col in df.columns]
    print("实际存在的所需列:", existing_columns)  # 打印实际存在的所需列

    if not existing_columns:
        print("输入文件中没有所需的列。")
        return

    # 提取需要的列
    result_df = df[existing_columns].copy()
    # rename_mapping = {
    #     'pm25_Model': 'model',
    #     'pm25_VNA': 'vna_pm25',
    #     'pm25_eVNA': 'evna_pm25',
    #     'pm25_aVNA': 'avna_pm25'
    # }
    rename_mapping = {
        'pm25_Model_mAverage': 'model',
        'pm25_VNA_mAverage': 'vna_pm25',
        'pm25_eVNA_mAverage': 'evna_pm25',
        'pm25_aVNA_mAverage': 'avna_pm25'
    }
    # 对存在的列进行重命名
    result_df.rename(columns={col: rename_mapping[col] for col in existing_columns if col in rename_mapping}, inplace=True)

    # 按照 _id 排序
    if '_id' in result_df.columns:
        result_df = result_df.sort_values(by=['_id']).reset_index(drop=True)

    # 在最终合并的数据表上计算 ROW 和 COL
    length = len(result_df)  # 获取最终数据表的长度
    rows_per_col = 246  # ROW 从 1 到 299
    cols = (length - 1) // rows_per_col + 1  # COL 最大值

    # 为整个数据表生成 ROW 和 COL
    rows = []
    cols_list = []
    for i in range(length):
        row = (i % rows_per_col) + 1  # ROW 从 1 开始，最大到 299
        col = (i // rows_per_col) + 1  # COL 从 1 开始，最大到 459
        cols_list.append(col)
        rows.append(row)

    # 将 ROW 和 COL 列添加到 result_df 中
    result_df['COL'] = cols_list
    result_df['ROW'] = rows

    # 重新按照 ROW 和 COL 排序
    result_df = result_df.sort_values(by=['ROW', 'COL']).reset_index(drop=True)

    # 将 _id、ROW 和 COL 列放到前面
    all_columns = ['_id', 'ROW', 'COL'] + [rename_mapping.get(col, col) for col in existing_columns if col != '_id']
    result_df = result_df[all_columns]

    # 保存处理后的数据
    result_df.to_csv(output_file_path, index=False)
    print(f"数据已成功处理并保存到 {output_file_path}")


# 输入文件路径
input_file = "/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/2017MAM_PM25_DFTAtF.csv"
# 输出文件路径
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/2017MAM_PM25_DFTAtF.csv'

# 调用函数处理文件
process_file(input_file, output_file)
