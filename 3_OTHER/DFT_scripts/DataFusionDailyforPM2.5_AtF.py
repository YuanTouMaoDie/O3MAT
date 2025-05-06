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
def start_period_averaged_data_fusion(model_file, monitor_file, region_table_file, file_path, dict_period,
                                      monitor_pollutant="pm25",
                                      model_pollutant="O3"):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} region_table_file: 包含 Is 列的数据表文件路径
    @param {string} file_path: 输出文件路径
    @param {dict} dict_period: 每个数据融合的时间段，键是时间段名称，值是时间段的开始和结束日期列表
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 pm25
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)  # 使用并行版本的NNA
    df_all_daily_prediction = None
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])

    # 读取包含 Is 列的数据表
    region_df = pd.read_csv(region_table_file)
    # 筛选出 Is 列值为 1 的行
    us_region_df = region_df[region_df['Is'] == 1]
    #-0.5确保后面的计算是正确的
    us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
    # 输入为x,y形式，对于model的col和row
    us_region_row_col = us_region_df[['COL', 'ROW']].values

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

            # 检查并处理 df_avg_obs 中 x 和 y 的重复值
            df_avg_obs["x"], df_avg_obs["y"] = proj(df_avg_obs["Lon"], df_avg_obs["Lat"])
            if df_avg_obs[['x', 'y']].duplicated().any():
                df_avg_obs = df_avg_obs.drop_duplicates(subset=['x', 'y'])

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
                # 检查时间索引的唯一性
                time_index = pd.date_range(start=start_date, end=end_date) if start_date <= end_date else \
                    pd.date_range(start=start_date, end='2017-12-31').union(pd.date_range(start='2017-01-01', end=end_date))
                if time_index.duplicated().any():
                    time_index = time_index.drop_duplicates()
                if start_date > end_date:
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(time_index)
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(dim="TSTEP")

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

            # 并行计算部分
            njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
            zdf = nn.predict(us_region_row_col, njobs=njobs)

            # 创建一个全为 NaN 的 DataFrame 用于存储预测结果
            result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_pm25", "vna_mod", "vna_bias", "vna_r_n"])

            # 将美国区域的预测结果填充到对应的行
            result_df.loc[us_region_df.index] = zdf

            df_prediction = pd.concat([df_prediction, result_df], axis=1)

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            df_fusion["avna_pm25"] = ds_avg_model[model_pollutant][0].values - df_fusion["vna_bias"]
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_avg_model[model_pollutant][0].shape)
            df_fusion["evna_pm25"] = (("ROW", "COL"), ds_avg_model[model_pollutant][0].values * reshaped_vna_r_n)
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
    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/3_OTHER/DFT_output/Region/Region_CONUS246396.csv"  # 替换为实际的包含 Is 列的数据表文件路径

    seasonal_output_path = os.path.join(save_path, "20170101_M25_PythonAtF.csv")
    start_period_averaged_data_fusion(
        model_file,
        monitor_file,
        region_table_file,
        seasonal_output_path,
        monitor_pollutant="Conc",
        model_pollutant="PM25_TOT",
        dict_period={
            "MAM": ["2017-01-01", "2017-01-01"],
        }
    )
    print("Done!")