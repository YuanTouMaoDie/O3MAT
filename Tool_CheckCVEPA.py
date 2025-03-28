import pandas as pd

# 定义文件路径
file_path_ds_input = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv'
file_path_ozone = '/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv'

try:
    # 读取两个 CSV 文件
    df_ds_input = pd.read_csv(file_path_ds_input)
    df_ozone = pd.read_csv(file_path_ozone)

    # 按 Date 列分组，获取每个日期对应的 Site 集合
    grouped_ds_input = df_ds_input.groupby('Date')['Site'].apply(set)
    grouped_ozone = df_ozone.groupby('Date')['Site'].apply(set)

    # 对比每个日期下的 Site 点
    comparison_result = {}
    for date in grouped_ozone.index:
        if date in grouped_ds_input.index:
            comparison_result[date] = grouped_ds_input[date].issuperset(grouped_ozone[date])
        else:
            comparison_result[date] = False

    # 输出结果
    for date, result in comparison_result.items():
        print(f"日期: {date}, ds.input 是否包含 ozone_2011 的 Site 点: {result}")

    # 检查是否存在 False 值
    has_false = False in comparison_result.values()
    if has_false:
        print("存在 ds.input 不包含 ozone_2011 的 Site 点的日期。")
        false_dates = [date for date, result in comparison_result.items() if not result]
        print("这些日期是:", false_dates)
    else:
        print("所有日期下，ds.input 都包含 ozone_2011 的 Site 点。")

except FileNotFoundError:
    print(f"错误: 文件未找到。请检查文件路径。")
except Exception as e:
    print(f"发生未知错误: {e}")
    