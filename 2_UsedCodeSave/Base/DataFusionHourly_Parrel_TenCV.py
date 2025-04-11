import pyrsig
import pyproj
import nna_methods  # 引入并行版本的NNA类
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
from esil.map_helper import show_maps
import cmaps
import xarray as xr
import multiprocessing  # 用于获取CPU核心数
import random

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_hourly_data_fusion(model_files, monitor_file, file_path, monitor_pollutant="ozone",
                             model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {list} model_files: 包含12个月模型数据的文件路径列表，每个文件对应一个月
    @param {string} monitor_file: 监测文件，必须有列：site_id, POCode, dateon, O3, Lat, Lon
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD HH:00'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD HH:00'
    """
    # 一次性读取所有模型文件
    ds_models = [pyrsig.open_ioapi(model_file) for model_file in model_files]
    df_obs = pd.read_csv(monitor_file)
    print("监测文件列名:", df_obs.columns)

    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    df_all_hourly_prediction = None

    df_obs['dateon'] = pd.to_datetime(df_obs['dateon'])
    df_obs_grouped = (
        df_obs.groupby(["site_id", "dateon"])
       .aggregate({"O3": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="dateon")
    ).reset_index()
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df_obs_grouped = df_obs_grouped[(df_obs_grouped["dateon"] >= start) & (df_obs_grouped["dateon"] <= end)]
    dates = df_obs_grouped["dateon"].dt.date.unique()  # 只取日期部分

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            df_hourly_obs = df_obs_grouped[df_obs_grouped["dateon"].dt.date == date].copy()

            # 检查数据框是否为空
            if df_hourly_obs.empty:
                print(f"警告：日期 {date} 的监测数据为空，跳过处理。")
                continue

            # 获取所有站点
            sites = df_hourly_obs["site_id"].unique()
            random.shuffle(sites)  # 随机打乱站点顺序

            # 分成十组
            num_splits = 10
            group_size = len(sites) // num_splits
            site_groups = [sites[i * group_size:(i + 1) * group_size].tolist() for i in range(num_splits)]
            # 处理剩余的站点
            remaining_sites = sites[num_splits * group_size:]
            for i, site in enumerate(remaining_sites):
                site_groups[i].append(site)

            for hour in range(24):
                current_date = pd.Timestamp(f"{date} {hour:02d}:00:00")
                month = current_date.month
                ds_model = ds_models[month - 1]
                proj = pyproj.Proj(ds_model.crs_proj4)

                tstep_value = pd.Timestamp(f"{current_date.year}-{month:02d}-{current_date.day:02d} {hour:02d}:00:00")
                try:
                    ds_hourly_model = ds_model.sel(TSTEP=tstep_value)
                except KeyError:
                    print(f"警告：日期 {current_date} 的模型数据缺失，跳过处理。")
                    continue

                df_current_hour_obs = df_hourly_obs.copy()
                df_current_hour_obs["x"], df_current_hour_obs["y"] = proj(df_current_hour_obs["Lon"], df_current_hour_obs["Lat"])

                # 记录匹配的模型网格点信息
                matched = ds_hourly_model[model_pollutant][0].sel(
                    ROW=df_current_hour_obs["y"].to_xarray(),
                    COL=df_current_hour_obs["x"].to_xarray(),
                    method="nearest",
                    drop=True
                )
                df_current_hour_obs["mod"] = matched
                df_current_hour_obs["ROW"] = matched.ROW.values
                df_current_hour_obs["COL"] = matched.COL.values

                for i in range(num_splits):
                    test_sites = site_groups[i]
                    train_sites = [site for group in site_groups[:i] + site_groups[i + 1:] for site in group]

                    df_train = df_current_hour_obs[df_current_hour_obs["site_id"].isin(train_sites)].copy()
                    df_test = df_current_hour_obs[df_current_hour_obs["site_id"].isin(test_sites)].copy()

                    # 检查训练集和测试集是否为空
                    if df_train.empty or df_test.empty:
                        print(f"警告：日期 {current_date} 的训练集或测试集为空，跳过此折处理。")
                        continue

                    # 计算训练集的 bias 和 r_n
                    df_train.loc[:, "bias"] = df_train["mod"] - df_train["O3"]
                    df_train.loc[:, "r_n"] = df_train["O3"] / df_train["mod"]

                    # 通过训练使得网格点能快速搜寻临近点
                    nn.fit(
                        df_train[["x", "y"]],
                        df_train[[monitor_pollutant, "mod", "bias", "r_n"]]
                    )

                    test_data = df_test[["COL", "ROW"]].values

                    # 并行计算，限定核心不能超过数据，否则报错
                    njobs = min(len(df_test), 88)
                    zdf = nn.predict(test_data, njobs=njobs)

                    # 将预测结果直接添加到 df_test 中
                    df_test.loc[:, ["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"]] = zdf

                    df_test.loc[:, "avna_ozone"] = ds_hourly_model[model_pollutant][0].sel(
                        ROW=df_test["y"].to_xarray(),
                        COL=df_test["x"].to_xarray(),
                        method="nearest"
                    ) - df_test["vna_bias"]

                    reshaped_vna_r_n = df_test["vna_r_n"].values
                    df_test.loc[:, "evna_ozone"] = ds_hourly_model[model_pollutant][0].sel(
                        ROW=df_test["y"].to_xarray(),
                        COL=df_test["x"].to_xarray(),
                        method="nearest"
                    ) * reshaped_vna_r_n

                    df_test.loc[:, "model"] = ds_hourly_model[model_pollutant][0].sel(
                        ROW=df_test["y"].to_xarray(),
                        COL=df_test["x"].to_xarray(),
                        method="nearest"
                    )

                    # 给 ROW 和 COL 加上 0.5
                    df_test.loc[:, "ROW"] = (df_test["site_id"].map(df_current_hour_obs.set_index("site_id")["ROW"]) + 0.5).astype(int)
                    df_test.loc[:, "COL"] = (df_test["site_id"].map(df_current_hour_obs.set_index("site_id")["COL"]) + 0.5).astype(int)

                    if df_all_hourly_prediction is None:
                        df_all_hourly_prediction = df_test
                    else:
                        df_all_hourly_prediction = pd.concat([df_all_hourly_prediction, df_test])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    if df_all_hourly_prediction is not None:
        df_all_hourly_prediction = df_all_hourly_prediction[["dateon", "site_id", "Lat", "Lon", "O3", "model", "vna_ozone", "evna_ozone", "avna_ozone", "ROW", "COL"]]
        df_all_hourly_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")
    else:
        print("警告：没有有效的预测数据，未生成输出文件。")

    return df_all_hourly_prediction


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_CV"
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

    # 指定日期范围
    start_date = '2011-01-01 00:00'
    end_date = '2011-01-01 23:00'

    hourly_output_path = os.path.join(save_path, "2011_SixDataset_CV_hourly.csv")
    start_hourly_data_fusion(
        model_files,
        monitor_file,
        hourly_output_path,
        monitor_pollutant="O3",
        model_pollutant="O3",
        start_date=start_date,
        end_date=end_date
    )
    print("Done!")
    