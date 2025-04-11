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

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, cross_validation_file, file_path, monitor_pollutant="ozone",
                            model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} cross_validation_file: 交叉验证文件，格式为 Date,Site,POC,Lat,Lon,Conc,CVgroup,Prediction,SEpred
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    @param {string} start_date: 开始日期，格式为 'YYYY-MM-DD'
    @param {string} end_date: 结束日期，格式为 'YYYY-MM-DD'
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    df_all_daily_prediction = None

    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    if start_date and end_date:
        df_obs_grouped = df_obs_grouped[(df_obs_grouped["Date"] >= start_date) & (df_obs_grouped["Date"] <= end_date)]
    dates = df_obs_grouped["Date"].unique()

    df_cv = pd.read_csv(cross_validation_file)
    # 读取 ds_ozone 数据
    df_ds_ozone = pd.read_csv(cross_validation_file)
    df_ds_ozone = df_ds_ozone[["Date", "Site", "Prediction"]].rename(columns={"Prediction": "ds_ozone"})

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
            df_daily_cv = df_cv[df_cv["Date"] == date].copy()

            # 提取和交叉验证表站点一样的数据
            df_daily_obs = df_daily_obs[df_daily_obs["Site"].isin(df_daily_cv["Site"])]

            if isinstance(ds_model['TSTEP'].values[0], np.int64):
                timeIndex = get_day_of_year(date) - 1
                ds_daily_model = ds_model.sel(TSTEP=timeIndex)
            else:
                ds_daily_model = ds_model.sel(TSTEP=date)

            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

            # 记录匹配的模型网格点信息
            matched = ds_daily_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest",
                drop=True
            )
            df_daily_obs["mod"] = matched
            df_daily_obs["ROW"] = matched.ROW.values
            df_daily_obs["COL"] = matched.COL.values

            cv_groups = df_daily_cv["CVgroup"].unique()
            for cv_group in cv_groups:
                test_sites = df_daily_cv[df_daily_cv["CVgroup"] == cv_group]["Site"]
                train_sites = df_daily_cv[df_daily_cv["CVgroup"] != cv_group]["Site"]

                df_train = df_daily_obs[df_daily_obs["Site"].isin(train_sites)]
                df_test = df_daily_obs[df_daily_obs["Site"].isin(test_sites)]

                # 计算训练集的 bias 和 r_n
                df_train.loc[:, "bias"] = df_train["mod"] - df_train["Conc"]
                df_train.loc[:, "r_n"] = df_train["Conc"] / df_train["mod"]

                # 通过训练使得网格点能快速搜寻临近点
                nn.fit(
                    df_train[["x", "y"]],
                    df_train[[monitor_pollutant, "mod", "bias", "r_n"]]
                )

                test_data = df_test[["COL", "ROW"]].values

                #并行计算，限定核心不能超过数据，否则报错
                njobs = min(len(df_test), 88)
                zdf = nn.predict(test_data, njobs=njobs)

                # 将预测结果直接添加到 df_test 中
                df_test[["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"]] = zdf

                df_test["avna_ozone"] = ds_daily_model[model_pollutant][0].sel(
                    ROW=df_test["y"].to_xarray(),
                    COL=df_test["x"].to_xarray(),
                    method="nearest"
                ) - df_test["vna_bias"]

                reshaped_vna_r_n = df_test["vna_r_n"].values
                df_test["evna_ozone"] = ds_daily_model[model_pollutant][0].sel(
                    ROW=df_test["y"].to_xarray(),
                    COL=df_test["x"].to_xarray(),
                    method="nearest"
                ) * reshaped_vna_r_n

                df_test["model"] = ds_daily_model[model_pollutant][0].sel(
                    ROW=df_test["y"].to_xarray(),
                    COL=df_test["x"].to_xarray(),
                    method="nearest"
                )

                # 给 ROW 和 COL 加上 0.5
                df_test["ROW"] = (df_test["Site"].map(df_daily_obs.set_index("Site")["ROW"]) + 0.5).astype(int)
                df_test["COL"] = (df_test["Site"].map(df_daily_obs.set_index("Site")["COL"]) + 0.5).astype(int)

                df_test["CVgroup"] = cv_group

                if df_all_daily_prediction is None:
                    df_all_daily_prediction = df_test
                else:
                    df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_test])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    # 合并 ds_ozone 数据
    df_all_daily_prediction = pd.merge(df_all_daily_prediction, df_ds_ozone, on=["Date", "Site"], how="left")

    # 调整 Site 顺序
    site_order = df_ds_ozone["Site"].unique()
    df_all_daily_prediction["Site"] = pd.Categorical(df_all_daily_prediction["Site"], categories=site_order, ordered=True)
    df_all_daily_prediction = df_all_daily_prediction.sort_values(by=["Date", "Site"])

    df_all_daily_prediction = df_all_daily_prediction[["Date", "Site", "Lat", "Lon", "Conc", "CVgroup","model","vna_ozone", "evna_ozone", "avna_ozone","ds_ozone", "ROW", "COL"]]
    df_all_daily_prediction.to_csv(file_path, index=False)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")
    return df_all_daily_prediction


# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_CV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    cross_validation_file = r"/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"  # 替换为实际的交叉验证文件路径

    # 指定日期范围
    start_date = '2011-01-01'
    end_date = '2011-12-31'

    daily_output_path = os.path.join(save_path, "2011_SixDataset_CV.csv")
    start_daily_data_fusion(
        model_file,
        monitor_file,
        cross_validation_file,
        daily_output_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        start_date=start_date,
        end_date=end_date
    )
    print("Done!")
    