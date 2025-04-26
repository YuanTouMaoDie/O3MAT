# import pandas as pd

# def check_sites_by_date(monitor_file, cross_validation_file, start_date=None, end_date=None):
#     """
#     按日期检查监测文件和交叉验证文件中的站点是否匹配
#     :param monitor_file: 监测文件路径
#     :param cross_validation_file: 交叉验证文件路径
#     :param start_date: 开始日期，格式为 'YYYY-MM-DD'
#     :param end_date: 结束日期，格式为 'YYYY-MM-DD'
#     :return: 匹配信息
#     """
#     try:
#         # 读取监测文件
#         df_monitor = pd.read_csv(monitor_file)
#         # 读取交叉验证文件
#         df_cv = pd.read_csv(cross_validation_file)

#         # 转换日期列为日期类型
#         df_monitor['Date'] = pd.to_datetime(df_monitor['Date'])
#         df_cv['Date'] = pd.to_datetime(df_cv['Date'])

#         # 根据日期范围筛选数据
#         if start_date and end_date:
#             start_date = pd.to_datetime(start_date)
#             end_date = pd.to_datetime(end_date)
#             df_monitor = df_monitor[(df_monitor['Date'] >= start_date) & (df_monitor['Date'] <= end_date)]
#             df_cv = df_cv[(df_cv['Date'] >= start_date) & (df_cv['Date'] <= end_date)]

#         # 按日期分组处理
#         unique_dates = sorted(set(df_monitor['Date'].tolist() + df_cv['Date'].tolist()))
#         for date in unique_dates:
#             monitor_data_on_date = df_monitor[df_monitor['Date'] == date]
#             cv_data_on_date = df_cv[df_cv['Date'] == date]

#             # 提取监测文件和交叉验证文件中的站点列
#             monitor_sites = set(monitor_data_on_date['Site'])
#             cv_sites = set(cv_data_on_date['Site'])

#             # 检查站点是否匹配
#             common_sites = monitor_sites.intersection(cv_sites)
#             only_in_monitor = monitor_sites - cv_sites
#             only_in_cv = cv_sites - monitor_sites

#             print(f"日期: {date.strftime('%Y-%m-%d')}")
#             print(f"监测文件和交叉验证文件中共同的站点数量: {len(common_sites)}")
#             print(f"仅存在于监测文件中的站点数量: {len(only_in_monitor)}")
#             if only_in_monitor:
#                 print("仅存在于监测文件中的站点:")
#                 print(only_in_monitor)
#             print(f"仅存在于交叉验证文件中的站点数量: {len(only_in_cv)}")
#             if only_in_cv:
#                 print("仅存在于交叉验证文件中的站点:")
#                 print(only_in_cv)
#             print("-" * 50)

#         return

#     except FileNotFoundError:
#         print("错误: 文件未找到，请检查文件路径。")
#     except KeyError:
#         print("错误: 文件中缺少 'Site' 或 'Date' 列，请检查文件格式。")
#     except Exception as e:
#         print(f"发生未知错误: {e}")

#     return None


# if __name__ == "__main__":
#     monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
#     cross_validation_file = r"/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"

#     # 指定日期范围
#     start_date = '2011-01-02'
#     end_date = '2011-01-02'

#     check_sites_by_date(monitor_file, cross_validation_file, start_date, end_date)


# import pandas as pd

# # 读取第一个文件
# df1 = pd.read_csv('/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv')
# # 读取第二个文件
# df2 = pd.read_csv('/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_equates_12km.csv')

# # 提取指定列并转换为字典列表
# dict_list1 = df1[['Date', 'Site', 'CVgroup']].to_dict('records')
# dict_list2 = df2[['Date', 'Site', 'CVgroup']].to_dict('records')

# # 比较两个字典列表是否完全相同
# if dict_list1 == dict_list2:
#     print("由Date、Site、CVgroup列构成的字典完全一样")
# else:
#     print("由Date、Site、CVgroup列构成的字典不完全一样")

import pandas as pd

# 定义文件路径
base_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126'
file1_path = f'{base_path}/2011_Monitor_W126_NotSameSiteWithSMAT.csv'
file2_path = f'{base_path}/2011_Monitor_W126.csv'

try:
    # 读取 CSV 文件
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 检查两个 DataFrame 中是否都存在 'Site' 列
    if 'Site' in df1.columns and 'Site' in df2.columns:
        # 获取第一个文件中的所有 Site
        sites_in_df1 = df1['Site'].unique()

        # 初始化计数器
        found_count = 0
        not_found_count = 0

        # 遍历每个 Site 并检查是否存在于第二个文件中
        for site in sites_in_df1:
            if site in df2['Site'].values:
                found_count += 1
            else:
                not_found_count += 1

        # 输出统计结果
        print(f"在 2011_Monitor_W126.csv 中找到的 Site 数量: {found_count}")
        print(f"在 2011_Monitor_W126.csv 中未找到的 Site 数量: {not_found_count}")
    else:
        print("其中一个文件中不存在 'Site' 列。")

except FileNotFoundError:
    print("错误：文件未找到，请检查文件路径是否正确。")
except Exception as e:
    print(f"发生未知错误：{e}")
    