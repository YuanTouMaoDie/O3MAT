import pandas as pd

# 定义文件路径
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/PM25ForFractions2002to2017.csv'

try:
    # 读取 CSV 文件，跳过第一行，并设置 low_memory=False 以处理混合数据类型
    df = pd.read_csv(file_path, skiprows=1, low_memory=False)

    # 检查 DATE 列是否存在
    if 'DATE' not in df.columns:
        print("错误: 数据框中不存在 'DATE' 列。请检查列名。")
    else:
        # 提取 2017 年的数据
        df_2017 = df[df['DATE'].astype(str).str.startswith('2017')]

        # 重命名列
        df_2017 = df_2017.rename(columns={'_ID': 'Site', 'LONG': 'Lon', 'LAT': 'Lat', 'PM25': 'Conc', 'DATE': 'Date'})

        # 剔除 USER_FLAG 列
        if 'USER_FLAG' in df_2017.columns:
            df_2017 = df_2017.drop(columns='USER_FLAG')

        # 剔除 EPA_FLAG 为 1 的行
        original_count = len(df_2017)
        df_2017 = df_2017[df_2017['EPA_FLAG'] != 1]
        removed_count = original_count - len(df_2017)
        print(f"剔除了 {removed_count} 行 EPA_FLAG 为 1 的数据。")

        # 过滤掉 pm25 < 0 的数据
        original_count_pm25 = len(df_2017)
        df_2017 = df_2017[df_2017['Conc'] >= 0]
        removed_count_pm25 = original_count_pm25 - len(df_2017)
        print(f"剔除了 {removed_count_pm25} 行 pm25 < 0 的数据。")

        # # 剔除 _TYPE 为 IMPROVE 的点
        # original_count_type = len(df_2017)
        # df_2017 = df_2017[df_2017['_TYPE'] != 'IMPROVE']
        # removed_count_type = original_count_type - len(df_2017)
        # print(f"剔除了 {removed_count_type} 行 _TYPE 为 IMPROVE 的数据。")

        # 检查是否有 Site 和 Date 重复的站点
        duplicate_rows = df_2017.duplicated(subset=['Site', 'Date'])
        if duplicate_rows.any():
            print("存在 Site 和 Date 重复的站点。")
        else:
            print("不存在 Site 和 Date 重复的站点。")

        # 转换日期格式
        df_2017['Date'] = pd.to_datetime(df_2017['Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # 保存为新的 CSV 文件
        output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/PM25ForFractions2017_1.csv'
        df_2017.to_csv(output_file, index=False)
        print(f"2017 年的数据已保存到 {output_file}")

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 未找到。")
except Exception as e:
    print(f"错误: 发生了一个未知错误: {e}")