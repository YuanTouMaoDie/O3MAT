import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator
import multiprocessing  # 用于获取CPU核心数


@timer_decorator
def start_hourly_data_fusion(model_files, monitor_file, region_table_file, file_path, monitor_pollutant="O3",
                             model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file: 监测文件，必须有列：site_id, POCode, dateon, dateoff, O3
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} lat_lon_file: 包含 site_id, Lat, Lon 信息的文件
    """
    # 处理日期范围
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    else:
        raise ValueError("start_date and end_date must be provided.")

    # 一次性读取所有模型文件
    ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]

    all_results = []

    # 遍历日期范围
    current_date = start
    while current_date <= end:
        month = current_date.month
        # 获取对应月份的模型数据
        ds_model = ds_models[month - 1]

        # 获取当天24小时的模型数据
        daily_model_data = []
        for hour in range(24):
            tstep_value = pd.Timestamp(f"{current_date.year}-{month:02d}-{current_date.day:02d} {hour:02d}:00:00")
            try:
                ds_hourly_model = ds_model.sel(TSTEP=tstep_value)
                daily_model_data.append(ds_hourly_model[model_pollutant][0].values.flatten())
            except KeyError:
                print(f"警告：日期 {tstep_value} 的模型数据缺失，跳过该小时处理。")
                daily_model_data.append(np.full_like(ds_model[model_pollutant][0].values.flatten(), np.nan))

        daily_model_data = np.array(daily_model_data)

        # 找出每天O3的最大值和最小值以及它们所属的时间段
        max_value = np.nanmax(daily_model_data, axis=0)
        min_value = np.nanmin(daily_model_data, axis=0)

        max_index = np.nanargmax(daily_model_data, axis=0)
        min_index = np.nanargmin(daily_model_data, axis=0)

        # 构建结果DataFrame
        df_result = pd.DataFrame({
            'date': current_date.strftime('%Y-%m-%d'),
            'O3_max': max_value,
            'O3_max_time': [f"{current_date.year}-{month:02d}-{current_date.day:02d} {int(index):02d}:00:00" for index in max_index],
            'O3_min': min_value,
            'O3_min_time': [f"{current_date.year}-{month:02d}-{current_date.day:02d} {int(index):02d}:00:00" for index in min_index]
        })
        all_results.append(df_result)

        current_date += pd.Timedelta(days=1)

    # 一次性合并所有结果
    df_all_results = pd.concat(all_results, ignore_index=True)

    df_all_results.to_csv(file_path, index=False)
    print(f"O3 max and min calculation for all dates is done, the results are saved to {file_path}")
    return df_all_results


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_files = [
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201101.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201102.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201103.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201104.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201105.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201106.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201107.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201108.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201109.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201110.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201111.nc",
        r"/backupdata/data_EPA/EQUATES/o3_hourly_files/EQUATES_COMBINE_ACONC_O3_201112.nc"
    ]

    monitor_file = r"/backupdata/data_EPA/aq_obs/routine/2011/AQS_hourly_data_2011_LatLon.csv"
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"
    lat_lon_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # 指定日期范围
    start_date = '2011/12/01 00:00'
    end_date = '2011/12/04 23:00'

    daily_output_path = os.path.join(save_path, "Test.csv")
    start_hourly_data_fusion(
        model_files,
        monitor_file,
        region_table_file,
        daily_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date,
    )
    print("Done!")