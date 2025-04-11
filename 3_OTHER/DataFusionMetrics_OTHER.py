import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator
from esil.map_helper import show_maps
import cmaps
import xarray as xr
import multiprocessing
import pyproj

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, save_path, variables):
    #定义投影信息
    proj_string = (
        "+proj=lcc "
        "+lat_0=30 +lon_0=112 "
        "+lat_1=25 +lat_2=40 "
        "+x_0=265500 +y_0=1089000 "
        "+R=6370000 "
        "+to_meter=3000 "
        "+no_defs"
    )
    proj = pyproj.Proj(proj_string)

    seasons = ['spring', 'summer', 'autumn', 'winter']
    # seasons = ['spring']
    for variable in variables:
        all_results = []
        for season in seasons:
            df_model = pd.read_csv(model_file)
            df_obs = pd.read_excel(monitor_file, sheet_name='Sheet1')
            nn = nna_methods.NNA(method="voronoi", k=30)
            df_all_daily_prediction = None

            # 根据输入的关键词匹配模型数据中的变量
            if season and variable:
                if "O3" in variable:
                    if "90" in variable:
                        model_col = f"O3_90-{season.capitalize()}"
                    else:
                        model_col = f"{variable}-{season.capitalize()}"
                else:
                    model_col = f"{variable}-{season.capitalize()}"
                if model_col not in df_model.columns:
                    raise ValueError(f"Column {model_col} not found in model file.")
                df_model = df_model[['_ID', 'LAT', 'LONG', 'ROW', 'COL', model_col]]
                model_pollutant = model_col
                df_model.rename(columns={model_col: model_pollutant}, inplace=True)

            # 根据输入的关键词匹配监测数据中的变量
            if season and variable:
                df_obs = df_obs[df_obs['time_range'] == season]
                if "O3" in variable:
                    if "90" in variable:
                        monitor_col = "O3-8H-90per"
                    else:
                        monitor_col = [col for col in df_obs.columns if "O3" in col and "D-MAX-MEAN" in col]
                        if not monitor_col:
                            raise ValueError(f"Column for O3-D-MAX-MEAN not found in monitor file.")
                        monitor_col = monitor_col[0]
                else:
                    monitor_col = variable
                if monitor_col not in df_obs.columns:
                    raise ValueError(f"Column {monitor_col} not found in monitor file.")
                df_obs = df_obs[['site_name', 'site_code', 'time_range', 'Lon', 'Lat', 'ROW', 'COL', monitor_col]]
                monitor_pollutant = monitor_col
                df_obs.rename(columns={monitor_col: monitor_pollutant}, inplace=True)

            df_daily_obs = df_obs.copy()

            #这边监测数据中已经附有其所在的模型网格点的坐标，因此不用nearest去查找
            df_daily_obs = pd.merge(df_daily_obs, df_model[['ROW', 'COL', model_pollutant]], on=['ROW', 'COL'], how='left')

            df_daily_obs["bias"] = df_daily_obs[model_pollutant] - df_daily_obs[monitor_pollutant]
            df_daily_obs["r_n"] = df_daily_obs[monitor_pollutant] / df_daily_obs[model_pollutant]
            
            # 将经纬度转换为模型的x, y坐标
            df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

            df_prediction = df_model[["ROW", "COL"]].copy()
            #找到网格中心点邻近的监测点
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, model_pollutant, "bias", "r_n"]]
            )

            njobs = multiprocessing.cpu_count()
            zdf = nn.predict(df_prediction[["COL", "ROW"]].values, njobs=njobs)

            # 统一输出变量名
            if variable == 'O3_90':
                df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T
            elif variable == 'O3':
                df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T
            else:
                df_prediction[f"vna_{variable}"] = zdf[:, 0]
                df_prediction["vna_mod"] = zdf[:, 1]
                df_prediction["vna_bias"] = zdf[:, 2]
                df_prediction["vna_r_n"] = zdf[:, 3]
                df_prediction.rename(columns={f"vna_{variable}": "vna_ozone"}, inplace=True)

            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
            rows = df_fusion.dims['ROW']
            cols = df_fusion.dims['COL']
            reshaped_model_values = df_model[model_pollutant].values.reshape(rows, cols)

            # 移除 avna_ozone 的计算
            # df_fusion["avna_ozone"] = reshaped_model_values - df_fusion["vna_bias"]

            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(rows, cols)
            df_fusion["evna_ozone"] = (("ROW", "COL"), reshaped_model_values * reshaped_vna_r_n)

            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = reshaped_model_values.flatten()
            df_fusion["Period"] = season
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
            df_fusion['Variable'] = variable
            all_results.append(df_fusion)

        final_result = pd.concat(all_results, ignore_index=True)
        output_file_name = f"Combined_AllSeasons_{variable}.csv"
        output_file_path = os.path.join(save_path, output_file_name)
        # 移除 avna_ozone 列
        final_result = final_result[['ROW', 'COL', 'Period', 'Variable', 'model', 'vna_ozone', 'evna_ozone']]
        final_result.to_csv(output_file_path, index=False)
        print(f"Data Fusion for all seasons and {variable} is done, the results are saved to {output_file_path}")


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/OTHER/Output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/OTHER/Input/Model.csv"
    monitor_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/OTHER/Input/季均统计数据_v01.xlsx"

    variables = ['O3_90', 'PM25']
    # variables = ['O3_90']

    start_daily_data_fusion(
        model_file,
        monitor_file,
        save_path,
        variables
    )
    print("Done!")