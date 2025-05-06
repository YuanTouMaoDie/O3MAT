# import pyrsig
# import pyproj

# model_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/CMAQ_daily_PM_O3_Species_2017.ioapi"
# ds_model = pyrsig.open_ioapi(model_file)
# proj = pyproj.Proj(ds_model.crs_proj4)
# print(ds_model.crs_proj4)


# import pandas as pd

# # 输入文件路径
# input_file = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/Region/246396_InUSA_Not.csv'

# try:
#     # 读取 CSV 文件
#     df = pd.read_csv(input_file)

#     # 检查每行是否都有值
#     df['Is'] = df.apply(lambda row: 1 if row.drop(['_id', 'ROW', 'COL']).notnull().all() else None, axis=1)

#     # 只保留 ROW, COL, Is 列
#     result_df = df[['ROW', 'COL', 'Is']]

#     # 输出文件路径
#     output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/Region/Region_CONUS246396.csv'

#     # 保存为 CSV 文件
#     result_df.to_csv(output_file, index=False)

#     print(f"处理完成，结果已保存到 {output_file}")

# except FileNotFoundError:
#     print(f"错误：未找到文件 {input_file}")
# except Exception as e:
#     print(f"发生未知错误: {e}")    