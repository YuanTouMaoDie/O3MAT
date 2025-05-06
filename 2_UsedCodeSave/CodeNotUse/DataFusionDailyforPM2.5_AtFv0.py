import pyrsig
import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
# for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps
import cmaps
import xarray as xr

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_period_averaged_data_fusion(model_file, monitor_file, file_path, dict_period, monitor_pollutant="ozone",
                                      model_pollutant="O3"):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: 输出文件路径
    @param {dict} dict_period: 每个数据融合的时间段，键是时间段名称，值是时间段的开始和结束日期列表
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])
    
    with tqdm(dict_period.items()) as pbar:
        for peroid_name, peroid in pbar:
            start_date, end_date = peroid[0], peroid[1]
            pbar.set_description(f"Data Fusion for {peroid_name}...")

            if start_date > end_date:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) | (df_obs["Date"] <= end_date)]
            else:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) & (df_obs["Date"] <= end_date)]

            df_avg_obs = (
                df_filtered_obs.groupby(["Site"]).aggregate(
                    {"Conc": "mean", "Lat": "mean", "Lon": "mean"}
                )
            ).reset_index()

            start_time = time.time()

            if isinstance(ds_model["TSTEP"].values[0], np.int64):
                start_time_index = get_day_of_year(start_date) - 1
                end_time_index = get_day_of_year(end_date) - 1
                if start_date > end_date:
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(range(start_time_index, len(ds_model["TSTEP"]))) + list(range(0, end_time_index + 1))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(
                        TSTEP=slice(start_time_index, end_time_index)
                    ).mean(dim="TSTEP")
            else:
                if start_date > end_date:
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(pd.date_range(start=start_date, end='2011-12-31')) + list(
                            pd.date_range(start='2011-01-01', end=end_date))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(dim="TSTEP")

            df_avg_obs["x"], df_avg_obs["y"] = proj(df_avg_obs["Lon"], df_avg_obs["Lat"])
            df_avg_obs["mod"] = ds_avg_model[model_pollutant][0].sel(
                ROW=df_avg_obs["y"].to_xarray(),
                COL=df_avg_obs["x"].to_xarray(),
                method="nearest"
            )
            df_avg_obs["bias"] = df_avg_obs["mod"] - df_avg_obs["Conc"]
            df_avg_obs["r_n"] = df_avg_obs["Conc"] / df_avg_obs["mod"]
            df_prediction = ds_avg_model[["ROW", "COL"]].to_dataframe().reset_index()
            nn.fit(
                df_avg_obs[["x", "y"]],
                df_avg_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
            )

            zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
            df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_ozone"] = ds_avg_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_avg_model[model_pollutant][0].shape)
            df_fusion["evna_ozone"] = (("ROW", "COL"), ds_avg_model[model_pollutant][0].values * reshaped_vna_r_n)
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_avg_model[model_pollutant][0].values.flatten()
            df_fusion["Period"] = peroid_name
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

            end_time = time.time()
            duration = end_time - start_time
            print(f"Data Fusion for {peroid_name} took {duration:.2f} seconds")

    df_all_daily_prediction.to_csv(file_path, index=False)
    print(f"Data Fusion for all dates is done, the results are saved to {file_path}")


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/CMAQ_daily_PM_O3_Species_2017.ioapi"
    monitor_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/PM25ForFractions2017.csv"

    seasonal_output_path = os.path.join(save_path, "2017_AnnualPM25_Python.csv")
    start_period_averaged_data_fusion(
        model_file,
        monitor_file,
        seasonal_output_path,
        monitor_pollutant="Conc",
        model_pollutant="PM25_TOT",
        dict_period={
            "DJF": ["2017-12-01", "2017-02-28"],
            "MAM": ["2017-03-01", "2017-05-31"],
            "JJA": ["2017-06-01", "2017-08-31"],
            "SON": ["2017-09-01", "2017-11-30"],
            "Annual": ["2017-01-01", "2017-12-31"],
            "Apr-Sep": ["2017-04-01", "2017-09-30"]
        }
    )
    print("Done!")