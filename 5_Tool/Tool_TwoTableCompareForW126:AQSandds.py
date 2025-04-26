#W126
# import pandas as pd

# # 读取文件
# file1_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_W126_SMAT_WithNaN.csv'
# file2_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_W126.csv'

# try:
#     df1 = pd.read_csv(file1_path)
#     df2 = pd.read_csv(file2_path)

#     # 检查两个 DataFrame 中是否都存在 'Site' 列
#     if 'Site' in df1.columns and 'Site' in df2.columns:
#         # 找出公共站点
#         common_sites = set(df1['Site']).intersection(set(df2['Site']))
#         print(f"公共站点数量: {len(common_sites)}")

#         # 筛选出公共站点的数据
#         df1_common = df1[df1['Site'].isin(common_sites)]
#         df2_common = df2[df2['Site'].isin(common_sites)]

#         # 统计公共站点中两边的无值站点（O3 列 NaN 或者 O3 < 0）
#         df1_common_invalid_count = 0
#         df2_common_invalid_count = 0
#         if 'O3' in df1_common.columns:
#             df1_common_invalid_count = len(df1_common[(df1_common['O3'].isna()) | (df1_common['O3'] < 0)])
#         if 'O3' in df2_common.columns:
#             df2_common_invalid_count = len(df2_common[(df2_common['O3'].isna()) | (df2_common['O3'] < 0)])

#         print(f"2011_Monitor_W126_SMAT_WithNaN.csv 中公共站点的无值站点数量: {df1_common_invalid_count}")
#         print(f"2011_Monitor_W126.csv 中公共站点的无值站点数量: {df2_common_invalid_count}")

#         # 找出公共站点中在两个文件里都无效的站点
#         if 'O3' in df1_common.columns and 'O3' in df2_common.columns:
#             # 合并公共站点数据
#             merged_common = pd.merge(df1_common[['Site', 'O3']], df2_common[['Site', 'O3']], on='Site', suffixes=('_1', '_2'))
#             # 筛选出在两个文件里都无效的站点
#             invalid_merged = merged_common[((merged_common['O3_1'].isna()) | (merged_common['O3_1'] < 0)) &
#                                            ((merged_common['O3_2'].isna()) | (merged_common['O3_2'] < 0))]
#             removed_count = len(invalid_merged)
#         else:
#             removed_count = 0

#         print(f"被剔除的公共站点中的无效站点数量: {removed_count}")

#         # 找出公共站点中一边有效一边无效的站点
#         if 'O3' in df1_common.columns and 'O3' in df2_common.columns:
#             # 筛选出 file1 有效 file2 无效的站点
#             one_valid_one_invalid_1 = merged_common[((merged_common['O3_1'] >= 0) & (~merged_common['O3_1'].isna())) &
#                                                     ((merged_common['O3_2'].isna()) | (merged_common['O3_2'] < 0))]
#             # 筛选出 file1 无效 file2 有效的站点
#             one_valid_one_invalid_2 = merged_common[((merged_common['O3_1'].isna()) | (merged_common['O3_1'] < 0)) &
#                                                     ((merged_common['O3_2'] >= 0) & (~merged_common['O3_2'].isna()))]
#             one_valid_one_invalid_count = len(one_valid_one_invalid_1) + len(one_valid_one_invalid_2)
#         else:
#             one_valid_one_invalid_count = 0

#         print(f"公共站点中一边有效一边无效的站点数量: {one_valid_one_invalid_count}")

#         # 筛选出两个文件中的有效站点
#         if 'O3' in df1.columns:
#             df1_valid = df1[(df1['O3'] >= 0) & (~df1['O3'].isna())]
#             df1_valid_count = len(df1_valid)
#             print(f"2011_Monitor_W126_SMAT_WithNaN.csv 中的有效站点数量: {df1_valid_count}")
#         if 'O3' in df2.columns:
#             df2_valid = df2[(df2['O3'] >= 0) & (~df2['O3'].isna())]
#             df2_valid_count = len(df2_valid)
#             print(f"2011_Monitor_W126.csv 中的有效站点数量: {df2_valid_count}")

#         # 找出有效公共站点
#         valid_common_sites = set(df1_valid['Site']).intersection(set(df2_valid['Site']))
#         print(f"有效公共站点数量: {len(valid_common_sites)}")

#         # 获取有效公共站点的交集数据
#         valid_common_result = df1_valid[df1_valid['Site'].isin(valid_common_sites)][['Site', 'Lat', 'Lon']]
#         valid_common_result['Flag'] = 1

#         # 找出 2011_Monitor_W126.csv 中有而 2011_MonitorW126_SMAT_WithNaN.csv 中没有的有效站点
#         df2_only_valid = df2_valid[~df2_valid['Site'].isin(df1_valid['Site'])]
#         df2_only_valid_result = df2_only_valid[['Site', 'Lat', 'Lon']]
#         df2_only_valid_result['Flag'] = 2

#         # 找出 2011_MonitorW126_SMAT_WithNaN.csv 中有而 2011_Monitor_W126.csv 中没有的有效站点
#         df1_only_valid = df1_valid[~df1_valid['Site'].isin(df2_valid['Site'])]
#         df1_only_valid_result = df1_only_valid[['Site', 'Lat', 'Lon']]
#         df1_only_valid_result['Flag'] = 3

#         # 合并所有有效结果
#         final_valid_result = pd.concat([valid_common_result, df2_only_valid_result, df1_only_valid_result])

#         # 统计 Flag 为 1、2、3 的个数
#         flag_1_count = len(final_valid_result[final_valid_result['Flag'] == 1])
#         flag_2_count = len(final_valid_result[final_valid_result['Flag'] == 2])
#         flag_3_count = len(final_valid_result[final_valid_result['Flag'] == 3])

#         print(f"Flag 为 1 的站点数量: {flag_1_count}")
#         print(f"Flag 为 2 的站点数量: {flag_2_count}")
#         print(f"Flag 为 3 的站点数量: {flag_3_count}")

#         # 保存结果到新的数据表
#         output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/2011_Monitor_SCUTEPAFlag.csv'
#         final_valid_result.to_csv(output_path, index=False)
#         print(f"结果已保存到 {output_path}")
#     else:
#         print("其中一个文件中不存在 'Site' 列。")

# except FileNotFoundError:
#     print("错误：文件未找到，请检查文件路径是否正确。")
# except Exception as e:
#     print(f"发生未知错误：{e}")

#AQS and daily
import pandas as pd

# 读取文件
file1_path = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv'
file2_path = '/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011_LatLon.csv'

try:
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df2['Site'] = df2['site_id']

    # 对各自的对应站点进行去重操作
    df1 = df1.drop_duplicates(subset=['Site'])
    df2 = df2.drop_duplicates(subset=['Site'])

    # 检查两个 DataFrame 中是否都存在 'Site' 列
    if 'Site' in df1.columns and 'Site' in df2.columns:
        # 找出公共站点
        common_sites = set(df1['Site']).intersection(set(df2['Site']))
        print(f"公共站点数量: {len(common_sites)}")

        # 筛选出公共站点的数据
        df1_common = df1[df1['Site'].isin(common_sites)]
        df2_common = df2[df2['Site'].isin(common_sites)]

        # 找出 AQS_hourly_data_2011_LatLon.csv 中有而 ds.input.aqs.o3.2011.csv 中没有的站点
        df2_only = df2[~df2['Site'].isin(df1['Site'])]
        df2_only_result = df2_only[['Site', 'Lat', 'Lon']]
        df2_only_result['Flag'] = 3

        # 找出 ds.input.aqs.o3.2011.csv 中有而 AQS_hourly_data_2011_LatLon.csv 中没有的站点
        df1_only = df1[~df1['Site'].isin(df2['Site'])]
        df1_only_result = df1_only[['Site', 'Lat', 'Lon']]
        df1_only_result['Flag'] = 2

        # 获取公共站点的交集数据
        valid_common_result = df1_common[['Site', 'Lat', 'Lon']]
        valid_common_result['Flag'] = 1

        # 合并所有结果
        final_result = pd.concat([valid_common_result, df2_only_result, df1_only_result])

        # 统计 Flag 为 1、2、3 的个数
        flag_1_count = len(final_result[final_result['Flag'] == 1])
        flag_2_count = len(final_result[final_result['Flag'] == 2])
        flag_3_count = len(final_result[final_result['Flag'] == 3])

        print(f"Flag 为 1 的站点数量: {flag_1_count}")
        print(f"Flag 为 2 的站点数量: {flag_2_count}")
        print(f"Flag 为 3 的站点数量: {flag_3_count}")

        # 保存结果到新的数据表
        output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_Monitor_Hourlvvs.DailyFlag.csv'
        final_result.to_csv(output_path, index=False)
        print(f"结果已保存到 {output_path}")
    else:
        print("其中一个文件中不存在 'Site' 列。")

except FileNotFoundError:
    print("错误：文件未找到，请检查文件路径是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")



