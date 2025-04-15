# import pandas as pd

# # 定义文件路径
# file_path_ds_input = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv'
# file_path_ozone = '/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv'

# try:
#     # 读取两个 CSV 文件
#     df_ds_input = pd.read_csv(file_path_ds_input)
#     df_ozone = pd.read_csv(file_path_ozone)

#     # 按 Date 列分组，获取每个日期对应的 Site 集合
#     grouped_ds_input = df_ds_input.groupby('Date')['Site'].apply(set)
#     grouped_ozone = df_ozone.groupby('Date')['Site'].apply(set)

#     # 对比每个日期下的 Site 点
#     comparison_result = {}
#     for date in grouped_ozone.index:
#         if date in grouped_ds_input.index:
#             comparison_result[date] = grouped_ds_input[date].issuperset(grouped_ozone[date])
#         else:
#             comparison_result[date] = False

#     # 输出结果
#     for date, result in comparison_result.items():
#         print(f"日期: {date}, ds.input 是否包含 ozone_2011 的 Site 点: {result}")

#     # 检查是否存在 False 值
#     has_false = False in comparison_result.values()
#     if has_false:
#         print("存在 ds.input 不包含 ozone_2011 的 Site 点的日期。")
#         false_dates = [date for date, result in comparison_result.items() if not result]
#         print("这些日期是:", false_dates)
#     else:
#         print("所有日期下，ds.input 都包含 ozone_2011 的 Site 点。")

# except FileNotFoundError:
#     print(f"错误: 文件未找到。请检查文件路径。")
# except Exception as e:
#     print(f"发生未知错误: {e}")


#臭氧曲线观察模拟
# import numpy as np
# import matplotlib.pyplot as plt


# def weighted_o3(o3):
#     return o3 / (1 + 4403 * np.exp(-126 * o3))


# # 初始化列表存储结果
# ozone_ppm = np.arange(0.001, 1, 0.001)
# monthly_w126 = []
# three_month_w126 = []

# # 计算每月和三个月的W126值
# for o3 in ozone_ppm:
#     daily_weighted = weighted_o3(o3) * 12
#     monthly = daily_weighted * 30
#     monthly_w126.append(monthly)
#     three_month = monthly * 3
#     three_month_w126.append(three_month)

# # 绘制每月W126值与臭氧浓度关系图
# plt.figure(figsize=(10, 6))
# plt.plot(ozone_ppm * 1000, monthly_w126, label='Monthly W126')
# plt.xlabel('Ozone Concentration (ppb)')
# plt.ylabel('Monthly W126 (ppm - hrs)')
# plt.title('Monthly W126 vs Ozone Concentration')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 绘制三个月W126值与臭氧浓度关系图
# plt.figure(figsize=(10, 6))
# plt.plot(ozone_ppm * 1000, three_month_w126, label='Three - Month W126')
# plt.xlabel('Ozone Concentration (ppb)')
# plt.ylabel('Three - Month W126 (ppm - hrs)')
# plt.title('Three - Month W126 vs Ozone Concentration')
# plt.legend()
# plt.grid(True)
# plt.show()


#从带 Is 的表中提取 Is = 1 的 ROW 和 COL，并将输入数据表中对应的 model 列设置为空值
import pandas as pd

try:
    # 读取带 Is 的表
    is_table_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv'
    df_is = pd.read_csv(is_table_path)

    # 读取输入数据表
    input_table_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/W126_VNA_2011_March_October.csv'
    df_input = pd.read_csv(input_table_path)

    # 从带 Is 的表中提取 Is = 1 的 ROW 和 COL
    filter_df = df_is[df_is['Is'] !=1][['ROW', 'COL']]

    # 根据提取的 ROW 和 COL 组合，将输入数据表中对应的 model 列设置为空值
    for _, row in filter_df.iterrows():
        df_input.loc[(df_input['ROW'] == row['ROW']) & (df_input['COL'] == row['COL']), 'vna_ozone'] = None

    # 保存修改后的数据表
    output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/W126_VNA_2011_March_October.csv'
    df_input.to_csv(output_path, index=False)
    print(f"已处理并保存到 {output_path}")

except FileNotFoundError:
    print("错误：未找到指定的 CSV 文件，请检查文件路径是否正确。")
except KeyError:
    print("错误：数据文件中缺少必要的列，请检查列名是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")
    
    
# #统计动态读取的EPA站点的数据行总数
# import pandas as pd

# def count_rows_in_csv(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         row_count = len(df)
#         return row_count
#     except FileNotFoundError:
#         print("错误: 文件未找到。")
#     except Exception as e:
#         print(f"错误: 发生了一个未知错误: {e}")
#     return None

# if __name__ == "__main__":
#     cross_validation_file = r"/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"
#     rows = count_rows_in_csv(cross_validation_file)
#     if rows is not None:
#         print(f"文件中的行数为: {rows}")    

    
    