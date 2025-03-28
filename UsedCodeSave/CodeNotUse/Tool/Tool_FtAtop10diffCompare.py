import pyrsig
import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
import xarray as xr
from esil.date_helper import timer_decorator, get_day_of_year


def get_top_value(model, model_pollutant, rank):
    """
    获取每个网格位置上污染物浓度降序排序后的第 rank 大的值。
    """
    pollutant_data = model[model_pollutant]

    def sort_and_take_top(arr, rank):
        sorted_arr = np.sort(arr)[::-1]
        return sorted_arr[rank - 1]  # rank 从 1 开始，所以索引为 rank - 1

    top_value = xr.apply_ufunc(
        sort_and_take_top,
        pollutant_data,
        input_core_dims=[['TSTEP']],
        kwargs={'rank': rank},
        vectorize=True
    )

    return top_value


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3_MDA8"):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: 输出文件路径
    @param {string} monitor_pollutant: 监测文件中的污染物，默认是 ozone
    @param {string} model_pollutant: 模型文件中的污染物，默认是 O3_MDA8
    """
    start_time = time.time()
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None

    # 挑选出各个站点中前10个浓度最高的日期
    df_obs_grouped = df_obs.groupby("Site")["Conc"].nlargest(10).reset_index()

    # 如果站点中该日期不足10天就剔除并显示剔除哪个站点Site
    site_counts = df_obs_grouped.groupby("Site").size()
    sites_to_remove = site_counts[site_counts < 10].index
    df_obs_grouped = df_obs_grouped[~df_obs_grouped["Site"].isin(sites_to_remove)]
    for site in sites_to_remove:
        print(f"剔除站点: {site}，因为该站点浓度最高的日期不足10天。")

    # 分别按所有站点第一高进行融合，其次第二高
    for rank in range(1, 11):  # rank 从 1 到 10
        # 获取当前排名对应的监测数据
        df_daily_obs = df_obs_grouped[df_obs_grouped.groupby("Site").cumcount() == rank - 1]
        df_daily_obs = df_daily_obs.merge(df_obs[["Site", "Lat", "Lon"]].drop_duplicates(), on="Site")

        # 获取模型数据中第 rank 大的值
        df_model_rank = get_top_value(ds_model, model_pollutant, rank)

        df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

        # 确保正确访问模型数据并处理索引
        df_daily_obs["mod"] = df_model_rank.sel(
            LAY=0,
            ROW=df_daily_obs["y"].to_xarray(),
            COL=df_daily_obs["x"].to_xarray(),
            method="nearest"
        )
        ds_model_rank_dataset = df_model_rank.to_dataset(name=model_pollutant)

        # 如果你想同时处理坐标，可以添加坐标信息作为数据变量
        ds_model_rank_dataset['ROW'] = ds_model_rank_dataset.ROW
        ds_model_rank_dataset['COL'] = ds_model_rank_dataset.COL

        # 现在可以像下面这样选取多个数据变量
        selected_ds = ds_model_rank_dataset[[model_pollutant, 'ROW', 'COL']]

        df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
        df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

        df_prediction = selected_ds[["ROW", "COL"]].to_dataframe().reset_index()
        nn.fit(
            df_daily_obs[["x", "y"]],
            df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
        )

        zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
        df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

        df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
        # 从 selected_ds 中选取 O3_MDA8 数据变量并获取其值
        o3_mda8_values = selected_ds[model_pollutant].values
        # 去除 LAY 维度（因为 LAY 只有一个元素）
        o3_mda8_values_2d = o3_mda8_values.squeeze()

        # 计算 avna_ozone 的值
        avna_ozone_values = o3_mda8_values_2d - df_fusion["vna_bias"].values
        # 明确指定维度名称
        df_fusion["avna_ozone"] = (('ROW', 'COL'), avna_ozone_values)

        reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(o3_mda8_values_2d.shape)
        df_fusion["evna_ozone"] = (('ROW', 'COL'), o3_mda8_values_2d * reshaped_vna_r_n)
        df_fusion = df_fusion.to_dataframe().reset_index()
        df_fusion["model"] = o3_mda8_values_2d.flatten()
        df_fusion["Period"] = f'top-{rank}_Fs'
        df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
        df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)

        if df_all_daily_prediction is None:
            df_all_daily_prediction = df_fusion
        else:
            df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

    # 最后得到结果再把结果进行平均
    df_all_daily_prediction = df_all_daily_prediction.groupby(["ROW", "COL"]).mean().reset_index()

    end_time = time.time()
    duration = end_time - start_time
    print(f"Data Fusion for all top values took {duration:.2f} seconds")

    if df_all_daily_prediction is not None:
        df_all_daily_prediction.to_csv(file_path, index=False)
        project_name = os.path.basename(file_path).replace(".csv", "")
        print(f"Data Fusion for all top values is done, the results are saved to {file_path}")
    else:
        print("没有有效的融合数据，未保存结果。")


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "BarronScript_ALL_2011_top10fs.csv"
    output_file_path = os.path.join(save_path, output_file_name)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    start_daily_data_fusion(
        model_file,
        monitor_file,
        output_file_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8"
    )