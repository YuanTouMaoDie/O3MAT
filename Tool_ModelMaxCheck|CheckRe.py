# import pyrsig
# import pyproj
# import nna_methods  # 引入并行版本的NNA类
# import os
# from tqdm.auto import tqdm
# import pandas as pd
# import time
# import numpy as np
# from esil.date_helper import timer_decorator
# import multiprocessing  # 用于获取CPU核心数


# @timer_decorator
# def check_model_date_range(model_files, start_date=None, end_date=None):
#     """
#     检查模型文件的日期范围
#     @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
#     @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
#     @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
#     """
#     # 处理日期范围
#     if start_date and end_date:
#         start = pd.to_datetime(start_date)
#         end = pd.to_datetime(end_date)
#     else:
#         raise ValueError("start_date and end_date must be provided.")

#     # 记录模型数据的实际开始和结束日期
#     actual_start_date = None
#     actual_end_date = None

#     # 一次性读取所有模型文件
#     ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]

#     # 遍历每个模型文件
#     for ds_model in ds_models:
#         try:
#             time_var = ds_model['TSTEP']
#             times = pd.to_datetime([time_var.attrs['units'].split(' ')[-1].strip('>')] + [t.decode('utf-8') for t in time_var[:].data], format='%Y%j%H%M%S')
#             if actual_start_date is None or times.min() < actual_start_date:
#                 actual_start_date = times.min()
#             if actual_end_date is None or times.max() > actual_end_date:
#                 actual_end_date = times.max()
#         except KeyError:
#             print("警告：模型文件中未找到时间变量 TSTEP，跳过该文件。")

#     print(f"模型数据的实际开始日期: {actual_start_date}")
#     print(f"模型数据的实际结束日期: {actual_end_date}")

#     if actual_start_date and actual_end_date:
#         if actual_start_date >= start and actual_end_date <= end:
#             print("模型数据的日期范围在指定范围内。")
#         else:
#             print("模型数据的日期范围不在指定范围内。")
#     else:
#         print("无法确定模型数据的完整日期范围。")


# # 在 main 函数中调用
# if __name__ == "__main__":
#     save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     model_files = [
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201101.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201102.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201103.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201104.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201105.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201106.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201107.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201108.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201109.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201110.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201111.nc",
#         r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201112.nc"
#     ]

#     # 指定日期范围
#     start_date = '2011/12/01 00:00'
#     end_date = '2011/12/04 23:00'

#     check_model_date_range(
#         model_files,
#         start_date=start_date,
#         end_date=end_date,
#     )
#     print("Done!")

import pandas as pd

# 读取 CSV 文件
file_path = 'output/Region/MonitorsTimeRegion_Filter.csv'
df = pd.read_csv(file_path)

# 检查是否存在重复行
if df.duplicated().any():
    print("数据中存在重复行。")
    # 打印重复行信息
    print("重复行信息如下：")
    print(df[df.duplicated()])
    # 删除重复行
    df = df.drop_duplicates()
    print("已删除重复行。")
else:
    print("数据中不存在重复行。")

# 保存处理后的数据到新的 CSV 文件
output_file_path = 'output/Region/MonitorsTimeRegion_Filter.csv'
df.to_csv(output_file_path, index=False)
print(f"处理后的数据已保存至 {output_file_path}。")