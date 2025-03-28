import pandas as pd


def merge_station_data(station_file, monitor_file, output_file, comparison_file):
    # 读取完整的监测数据，并过滤掉 O3 为 -999 的行
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    original_monitor_site_ids = df_monitor['site_id'].unique()  # 去除前的监测数据站点
    df_monitor = df_monitor[df_monitor['O3'] != -999]  # 去除 O3 为 -999 的行
    filtered_monitor_site_ids = df_monitor['site_id'].unique()  # 去除后的监测数据站点

    # 读取站点数据，获取唯一站点 ID
    df_station = pd.read_csv(station_file, header=None, usecols=[0], names=['Site'], dtype={'Site': str})
    original_station_site_ids = df_station['Site'].unique()  # 站点数据中的唯一站点
    filtered_station_site_ids = df_station['Site'].unique()  # 假设站点文件不含 -999，保持不变

    # 打印去除 O3 == -999 后，站点数量的变化
    print(f"去除 O3 == -999 后，监测文件中的唯一站点数量：{len(filtered_monitor_site_ids)}")
    print(f"站点文件中的唯一站点数量：{len(filtered_station_site_ids)}")

    # 比较原始和过滤后的唯一站点 ID
    print(f"原始监测文件中的站点数量：{len(original_monitor_site_ids)}")
    print(f"原始站点文件中的站点数量：{len(original_station_site_ids)}")

    # 继续读取监测数据（已经过滤了 O3 为 -999 的行）
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    df_monitor = df_monitor[df_monitor['site_id'].isin(filtered_monitor_site_ids)]

    # 读取站点数据，但只保留唯一站点对应的数据，指定数据类型为字符串
    df_station = pd.read_csv(station_file, header=None,
                             names=['Site', 'POC', 'Date', 'Lat', 'Lon', 'Conc'],
                             dtype={'Site': str})
    df_station = df_station[df_station['Site'].isin(filtered_station_site_ids)]

    # 对于每个站点，取第一个出现的经纬度信息
    df_station = df_station.groupby('Site').first().reset_index()

    # 合并数据，根据站点 ID
    merged_df = pd.merge(df_monitor, df_station, left_on='site_id',
                         right_on='Site', how='left')

    # 选择需要的列
    final_df = merged_df[['site_id', 'POCode', 'dateon', 'dateoff', 'O3', 'Lat', 'Lon']]

    # 去除没有经纬度的站点
    final_df = final_df.dropna(subset=['Lat', 'Lon'])

    # 保存最终结果
    final_df.to_csv(output_file, index=False)
    print("数据合并完成，结果已保存到", output_file)

    # 比较缺失和多出的站点
    missing_sites = set(filtered_station_site_ids) - set(filtered_monitor_site_ids)
    extra_sites = set(filtered_monitor_site_ids) - set(filtered_station_site_ids)

    # 将比较结果保存到 CSV
    comparison_df = pd.DataFrame({
        'Missing Site ID in merged data': list(missing_sites),
        'Extra Site ID in merged data': list(extra_sites)
    })
    comparison_df.to_csv(comparison_file, index=False)
    print(f"缺失和多余的站点信息已保存到 {comparison_file}")


if __name__ == "__main__":
    monitor_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/AQS_hourly_data_2011.csv"
    station_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    output_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/AQS_hourly_data_2011_LatLon.csv"
    comparison_file = r"/backupdata/data_EPA/EQUATES/2011_Hour_Data/comparison_missing_extra_sites.csv"

    # 读取监测文件并过滤 O3 == -999 后的站点 ID
    df_monitor = pd.read_csv(monitor_file, dtype={'site_id': str})
    df_monitor_filtered = df_monitor[df_monitor['O3'] != -999]
    filtered_monitor_site_ids = df_monitor_filtered['site_id'].unique()

    # 读取站点文件并获取唯一站点 ID
    df_station = pd.read_csv(station_file, header=None, usecols=[0], names=['Site'], dtype={'Site': str})
    filtered_station_site_ids = df_station['Site'].unique()

    # 打印去除 O3 == -999 后站点数量
    print(f"去除 O3 == -999 后，监测文件中的唯一站点数量：{len(filtered_monitor_site_ids)}")
    print(f"站点文件中的唯一站点数量：{len(filtered_station_site_ids)}")

    # 找出站点文件中存在但监测文件中没有的站点
    missing_in_monitor_file = set(filtered_station_site_ids) - set(filtered_monitor_site_ids)
    extra_in_monitor_file = set(filtered_monitor_site_ids) - set(filtered_station_site_ids)

    if missing_in_monitor_file or extra_in_monitor_file:
        print("缺失的站点：", missing_in_monitor_file)
        print("多余的站点：", extra_in_monitor_file)

        # 将缺失的站点和多余的站点保存到文件
        comparison_df = pd.DataFrame({
            'Missing Site ID in merged data': list(missing_in_monitor_file),
            'Extra Site ID in merged data': list(extra_in_monitor_file)
        })
        comparison_df.to_csv(comparison_file, index=False)
        print(f"缺失和多余的站点信息已保存到 {comparison_file}")
    else:
        print("没有缺失或多余的站点。")

    # 继续执行数据合并过程
    merge_station_data(station_file, monitor_file, output_file, comparison_file)
