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
def start_daily_data_fusion(model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3", start_date=None, end_date=None):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
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

    # 根据指定的日期范围筛选数据
    if start_date and end_date:
        df_obs_grouped = df_obs_grouped[(df_obs_grouped["Date"] >= start_date) & (df_obs_grouped["Date"] <= end_date)]

    dates = df_obs_grouped["Date"].unique()

    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            start_time = time.time()

            df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
            if isinstance(ds_model['TSTEP'].values[0], np.int64):
                timeIndex = get_day_of_year(date) - 1
                ds_daily_model = ds_model.sel(TSTEP=timeIndex)
            else:
                ds_daily_model = ds_model.sel(TSTEP=date)

            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])
            df_daily_obs["mod"] = ds_daily_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest"
            )
            df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
            df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

            df_prediction = ds_daily_model[["ROW", "COL"]].to_dataframe().reset_index()
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            # 并行计算部分
            njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
            zdf = nn.predict(df_prediction[["COL", "ROW"]].values, njobs=njobs)
            df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_daily_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_daily_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_daily_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_daily_model[model_pollutant][0].values.flatten()
            df_fusion["Timestamp"] = date
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {date} took {duration:.2f} seconds")

    df_all_daily_prediction.to_csv(file_path, index=False)
    project_name = os.path.basename(file_path).replace(".csv", "")
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

# 在 main 函数中调用
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "Test"
    output_file_path = os.path.join(save_path, output_file_name)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    # model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # 指定日期范围
    start_date = '2011-01-01'
    end_date = '2011-01-10'

    daily_output_path = os.path.join(save_path, "Test_OnlyParrel.csv")
    start_daily_data_fusion(
        model_file,
        monitor_file,
        daily_output_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        start_date=start_date,
        end_date=end_date
    )
    print("Done!")
    