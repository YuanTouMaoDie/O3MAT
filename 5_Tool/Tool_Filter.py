import pandas as pd


def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("错误：未找到指定的 CSV 文件，请检查文件路径是否正确。")
        return None


def filter_data(df_is, df_input):
    try:
        # 从带 Is 的表中提取 Is = 1 的 ROW 和 COL
        filter_df = df_is[df_is['Is'] != 1][['ROW', 'COL']]
        # 使用 merge 函数进行条件筛选
        merged = pd.merge(df_input, filter_df, on=['ROW', 'COL'], how='left', indicator=True)
        df_input.loc[merged['_merge'] == 'both', 'model'] = None
        return df_input
    except KeyError:
        print("错误：数据文件中缺少必要的列，请检查列名是否正确。")
        return None


def save_csv_file(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"已处理并保存到 {output_path}")
    except Exception as e:
        print(f"保存文件时发生错误：{e}")


if __name__ == "__main__":
    # 读取带 Is 的表
    is_table_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv'
    df_is = read_csv_file(is_table_path)
    if df_is is None:
        exit(1)

    # 读取输入数据表
    input_table_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/W126_CMAQ_2011_March_October.csv'
    df_input = read_csv_file(input_table_path)
    if df_input is None:
        exit(1)

    # 过滤数据
    df_filtered = filter_data(df_is, df_input)
    if df_filtered is not None:
        # 保存修改后的数据表
        output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/W126_CMAQ_2011_March_October_CONUS.csv'
    
