import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps
import cmaps

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


def plot_us_map(
    fusion_output_file,
    model_file,
    save_path=None,
    boundary_json_file="/DeepLearning/mnt/Devin/boundary/USA_State.json",
):
    """
    description: plot the US map with different data fusion results
    @param {string} fusion_output_file: the data fusion output file
    @param {string} model_file: the model file used for data fusion
    @param {string} save_path: the path to save the plot
    @param {string} boundary_json_file: the boundary file for the US map
    @return None
    """
    mp = model_attribute(model_file)
    proj, longitudes, latitudes = mp.projection, mp.lons, mp.lats
    df_data = pd.read_csv(fusion_output_file)
    if "Period" not in df_data.columns:
        print("The data fusion file does not contain the Period column!")
        return
    layout = None
    periods = df_data["Period"].unique()
    for period in tqdm(periods):
        dict_data = {}
        df_period = df_data[df_data["Period"] == period]
        grid_concentration_model = df_period["model"].values.reshape(longitudes.shape)
        grid_concentration_vna_ozone = df_period["vna_ozone"].values.reshape(
            longitudes.shape
        )
        grid_concentration_evna_ozone = df_period["evna_ozone"].values.reshape(
            longitudes.shape
        )
        grid_concentration_avna_ozone = df_period["avna_ozone"].values.reshape(
            longitudes.shape
        )
        period = (
            period.replace("_daily_DF", "")
            .replace("average", "Avg.")
            .replace("Average", "Avg.")
        )
        vmax_conc = max(
            np.nanpercentile(grid_concentration_model, 99.5),
            np.nanpercentile(grid_concentration_vna_ozone, 99.5),
            np.nanpercentile(grid_concentration_evna_ozone, 99.5),
            np.nanpercentile(grid_concentration_avna_ozone, 99.5),
        )
        vmin_conc = min(
            np.nanpercentile(grid_concentration_model, 0.5),
            np.nanpercentile(grid_concentration_vna_ozone, 0.5),
            np.nanpercentile(grid_concentration_evna_ozone, 0.5),
            np.nanpercentile(grid_concentration_avna_ozone, 0.5),
        )
        value_range = (vmin_conc, vmax_conc)
        # get_multiple_data (
        #     dict_data,
        #     dataset_name=f"{period}_EQUATES",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_model,
        # )
        get_multiple_data(
            dict_data,
            dataset_name=f"{period}_vna_ozone",
            variable_name="",
            grid_x=longitudes,
            grid_y=latitudes,
            grid_concentration=grid_concentration_vna_ozone,
        )
        # get_multiple_data(
        #     dict_data,
        #     dataset_name=f"Delta (vna_ozone - EQUATES)",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_vna_ozone - grid_concentration_model,
        #     is_delta=True,
        #     cmap=cmap_delta,
        # )
        fig = show_maps(
            dict_data,
            unit="ppbv",
            cmap=cmap_conc,
            show_lonlat=True,
            projection=proj,
            is_wrf_out_data=True,
            boundary_file=boundary_json_file,
            show_original_grid=True,
            panel_layout=layout,
            delta_map_settings={
                "cmap": cmap_delta,
                "value_range": (None,None),
                "colorbar_ticks_num": None,
                "colorbar_ticks_value_format": ".2f",
                "value_format": ".2f",
            },
            title_fontsize=11,
            xy_title_fontsize=9,
            show_dependenct_colorbar=True,
            value_range=value_range,
        )

        if save_path is not None:
            save_file = os.path.join(save_path, f"{period}_vna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period} is saved to {save_file}")
        dict_data = {}

        # get_multiple_data(
        #     dict_data,
        #     dataset_name=f"{period}_EQUATES",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_model,
        # )
        get_multiple_data(
            dict_data,
            dataset_name=f"{period}_evna_ozone",
            variable_name="",
            grid_x=longitudes,
            grid_y=latitudes,
            grid_concentration=grid_concentration_evna_ozone,
        )
        # get_multiple_data(
        #     dict_data,
        #     dataset_name=f"Delta (evna_ozone - EQUATES)",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_evna_ozone - grid_concentration_model,
        #     is_delta=True,
        #     cmap=cmap_delta,
        # )
        fig = show_maps(
            dict_data,
            unit="ppbv",
            cmap=cmap_conc,
            show_lonlat=True,
            projection=proj,
            is_wrf_out_data=True,
            boundary_file=boundary_json_file,
            show_original_grid=True,
            panel_layout=layout,
            delta_map_settings={
                "cmap": cmap_delta,
                "value_range": (None, None),
                "colorbar_ticks_num": None,
                "colorbar_ticks_value_format": ".2f",
                "value_format": ".2f",
            },
            title_fontsize=11,
            xy_title_fontsize=9,
            show_dependenct_colorbar=True,
            value_range=value_range,
        )
        if save_path is not None:
            save_file = os.path.join(save_path, f"{period}_evna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period} is saved to {save_file}")
        dict_data = {}
        # get_multiple_data(
        #     dict_data,
        #     dataset_name=f"{period}_EQUATES",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_model,
        # )
        get_multiple_data(
            dict_data,
            dataset_name=f"{period}_avna_ozone",
            variable_name="",
            grid_x=longitudes,
            grid_y=latitudes,
            grid_concentration=grid_concentration_evna_ozone,
        )
        # get_multiple_data(
        #     dict_data,
        #     dataset_name=f"Delta (evna_ozone - EQUATES)",
        #     variable_name="",
        #     grid_x=longitudes,
        #     grid_y=latitudes,
        #     grid_concentration=grid_concentration_evna_ozone - grid_concentration_model,
        #     is_delta=True,
        #     cmap=cmap_delta,
        # )

        # layout = (3, 4) if len(dict_data) == 12 else None
        # fig = show_maps(
        #     dict_data,
        #     unit="ppbv",
        #     cmap=cmap_conc,
        #     show_lonlat=True,
        #     projection=proj,
        #     is_wrf_out_data=True,
        #     boundary_file=boundary_json_file,
        #     show_original_grid=True,
        #     panel_layout=layout,
        #     delta_map_settings={
        #         "cmap": cmap_delta,
        #         "value_range": (None, None),
        #         "colorbar_ticks_num": None,
        #         "colorbar_ticks_value_format": ".2f",
        #         "value_format": ".2f",
        #     },
        #     title_fontsize=11,
        #     xy_title_fontsize=9,
        #     show_dependenct_colorbar=True,
        #     value_range=value_range,
        # )
        # if save_path is not None:
        #     save_file = os.path.join(save_path, f"{period}_avna_ozone_data_fusion.png")
        #     fig.savefig(save_file, dpi=300)
        #     print(f"The data fusion plot for {period} is saved to {save_file}")


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    data_fusion_file =r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_seasonal_O3-MDA8.csv"
    # file_list=save_daily_data_fusion_to_metrics(data_fusion_file,save_path,project_name="2011_daily_DF")
    file_list = [
        # "/DeepLearning/mnt/Devin/data_fusion/output/2011_daily_DF_98th percentile O3-MDA8.csv",
        # "/DeepLearning/mnt/Devin/data_fusion/output/2011_daily_DF_Average of top-10 O3-MDA8 days.csv",
        "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_FtA_Python_filteredForMap_ROWCOL_JSON_Special.csv"
        # "/DeepLearning/mnt/Devin/data_fusion/output/2011_daily_DF_Summer season average (Apr-Sep) of O3-MDA8.csv",
        # "/DeepLearning/mnt/Devin/data_fusion/output/2011_daily_DF_seasonal_O3-MDA8.csv",
        #         "/DeepLearning/mnt/Devin/data_fusion/output/2011_daily_DF_seasonal_O3-MDA8.csv",
    ]
    save_path = os.path.join(save_path, "Zhuo")
    for file in file_list:
        plot_us_map(file, model_file, save_path)
    print("Done!")
