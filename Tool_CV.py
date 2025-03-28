import pandas as pd

def check_sites_by_date(monitor_file, cross_validation_file, start_date=None, end_date=None):
    """
    按日期检查监测文件和交叉验证文件中的站点是否匹配
    :param monitor_file: 监测文件路径
    :param cross_validation_file: 交叉验证文件路径
    :param start_date: 开始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :return: 匹配信息
    """
    try:
        # 读取监测文件
        df_monitor = pd.read_csv(monitor_file)
        # 读取交叉验证文件
        df_cv = pd.read_csv(cross_validation_file)

        # 转换日期列为日期类型
        df_monitor['Date'] = pd.to_datetime(df_monitor['Date'])
        df_cv['Date'] = pd.to_datetime(df_cv['Date'])

        # 根据日期范围筛选数据
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df_monitor = df_monitor[(df_monitor['Date'] >= start_date) & (df_monitor['Date'] <= end_date)]
            df_cv = df_cv[(df_cv['Date'] >= start_date) & (df_cv['Date'] <= end_date)]

        # 按日期分组处理
        unique_dates = sorted(set(df_monitor['Date'].tolist() + df_cv['Date'].tolist()))
        for date in unique_dates:
            monitor_data_on_date = df_monitor[df_monitor['Date'] == date]
            cv_data_on_date = df_cv[df_cv['Date'] == date]

            # 提取监测文件和交叉验证文件中的站点列
            monitor_sites = set(monitor_data_on_date['Site'])
            cv_sites = set(cv_data_on_date['Site'])

            # 检查站点是否匹配
            common_sites = monitor_sites.intersection(cv_sites)
            only_in_monitor = monitor_sites - cv_sites
            only_in_cv = cv_sites - monitor_sites

            print(f"日期: {date.strftime('%Y-%m-%d')}")
            print(f"监测文件和交叉验证文件中共同的站点数量: {len(common_sites)}")
            print(f"仅存在于监测文件中的站点数量: {len(only_in_monitor)}")
            if only_in_monitor:
                print("仅存在于监测文件中的站点:")
                print(only_in_monitor)
            print(f"仅存在于交叉验证文件中的站点数量: {len(only_in_cv)}")
            if only_in_cv:
                print("仅存在于交叉验证文件中的站点:")
                print(only_in_cv)
            print("-" * 50)

        return

    except FileNotFoundError:
        print("错误: 文件未找到，请检查文件路径。")
    except KeyError:
        print("错误: 文件中缺少 'Site' 或 'Date' 列，请检查文件格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")

    return None


if __name__ == "__main__":
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    cross_validation_file = r"/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"

    # 指定日期范围
    start_date = '2011-01-02'
    end_date = '2011-01-02'

    check_sites_by_date(monitor_file, cross_validation_file, start_date, end_date)
