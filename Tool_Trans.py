import pandas as pd


def transform_data(df):
    # 从 Period 列提取实际的 Period 值
    df['Period'] = df['Period'].str.split('_').str[0]
    # 按 ROW、COL 和 Period 分组，对各臭氧列进行聚合填充
    grouped = df.groupby(['ROW', 'COL', 'Period'])
    result = grouped.agg({
        'vna_ozone': 'first',
        'evna_ozone': 'first',
        'avna_ozone': 'first',
        'model': 'first'
    }).reset_index()
    # 按 Period、ROW、COL 进行多级排序
    result = result.sort_values(by=['Period', 'ROW', 'COL'])
    return result


if __name__ == "__main__":
    # 读取 CSV 文件
    file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_HourlyMetrics_metrics.csv'
    try:
        df = pd.read_csv(file_path)
        # 调用函数进行数据转换
        transformed_df = transform_data(df)
        # 保存转换后的数据
        output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_HourlyMetrics_Trans.csv'
        transformed_df.to_csv(output_file, index=False)
        print(f"数据转换完成，结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    