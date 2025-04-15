# import pandas as pd


# def transform_data(df):
#     # 从 Period 列提取实际的 Period 值
#     df['Period'] = df['Period'].str.split('_').str[0]
#     # 按 ROW、COL 和 Period 分组，对各臭氧列进行聚合填充
#     grouped = df.groupby(['ROW', 'COL', 'Period'])
#     result = grouped.agg({
#         'vna_ozone': 'first',
#         'evna_ozone': 'first',
#         'avna_ozone': 'first',
#         'model': 'first'
#     }).reset_index()
#     # 按 Period、ROW、COL 进行多级排序
#     result = result.sort_values(by=['Period', 'ROW', 'COL'])
#     return result


# if __name__ == "__main__":
#     # 读取 CSV 文件
#     file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_W126_TrueTrue.csv'
#     try:
#         df = pd.read_csv(file_path)
#         # 调用函数进行数据转换
#         transformed_df = transform_data(df)
#         # 保存转换后的数据
#         output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_W126_TrueTrue_Trans.csv'
#         transformed_df.to_csv(output_file, index=False)
#         print(f"数据转换完成，结果已保存到 {output_file}")
#     except FileNotFoundError:
#         print(f"错误：未找到文件 {file_path}")
#     except Exception as e:
#         print(f"发生未知错误: {e}")
    
# import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('output/2011_Data_CV/2011_SixDataset_CV_Metrics.csv')

# # 修改Period列的值
# df['Period'] = df['Period'].replace('Apr - Sep', 'Apr-Sep')
# df['Period'] = df['Period'].replace('top - 10', 'top-10')

# # 保存修改后的数据表
# df.to_csv('output/2011_Data_CV/2011_SixDataset_CV_Metrics.csv', index=False)

# print("数据处理完成，已保存为2011_SixDataset_CV_Metrics_modified.csv")
import pandas as pd
import math

try:
    df = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_ROWCOLRegion_Tz_CONUS_DT.csv')
    if 'gmt_offset' in df.columns:
        def subtract_one(x):
            if pd.isna(x) or math.isnan(x):
                return x
            try:
                num = int(x)
                return num - 1
            except ValueError:
                raise ValueError(f"值 '{x}' 无法转换为整数，请检查数据。")

        df['gmt_offset'] = df['gmt_offset'].apply(subtract_one)
        new_file_name = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_ROWCOLRegion_Tz_CONUS_ST.csv'
        df.to_csv(new_file_name, index=False)
        print(f"gmt_offset 列处理完成，结果已保存到 {new_file_name}。")
    else:
        print("文件中不存在 gmt_offset 列。")

except FileNotFoundError:
    print("未找到指定的 CSV 文件。")
except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"发生未知错误: {e}")
    
    
    
    
    