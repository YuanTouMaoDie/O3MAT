import os
import re
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

# for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps
import cmaps

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


def extract_key_period(period):
    """
    Extract key period (e.g., JFM, AMJ) from the full period string.
    """
    key_periods = ["DJF", "MAM", "JJA", "SON", 'Annual', 'Apr-Sep', 'top-10', '98th']  # Add more if needed
    for key in key_periods:
        if key in period:
            return key
    return None


def get_prefix(filename):
    """
    Determine the prefix based on the filename.
    If the filename contains 'daily' or 'IA', return 'IA'; otherwise, return 'AI'.
    """
    if "daily" in filename.lower() or "FtA" in filename:
        return "FtA"
    return "AtF"


def get_year(filename):
    """
    Extract the year from the filename (assuming the year is in the range 2011-2020).
    """
    match = re.search(r"(20[1-2][0-9])", filename)  # Match years between 2011 and 2020
    if match:
        return match.group(1)
    return None


def get_axis_label(filename, period=None, year=None):
    """
    Generate the axis label based on the filename.
    - If the filename contains 'DFT', the label is 'DFT'.
    - If the filename does not contain 'DFT', the label is 'Python'.
    - Combine with the prefix (IA or AI), period, and year to form the final label.
    """
    prefix = get_prefix(filename)
    if "DFT" in filename.upper():
        label = "DFT"
    elif "BarronResult" in filename:
        label = "Barron's Result"
    elif "BarronScript" in filename:
        label = "Barron's Script"
    elif "Python" in filename:
        label = "Python"
    elif "EQUATES" in filename:
        label = "EQUATES"
    else:
        label = "unkown"

    # Add period and year to the label if provided
    if period and year:
        return f"{prefix}_{label}_{period}"
    elif period:
        return f"{prefix}_{label}_{period}"
    elif year:
        return f"{prefix}_{label}_{year}"
    return f"{prefix}_{label}"


def plot_us_map(
        fusion_output_file,
        model_file,
        base_save_path=None,
        boundary_json_file="/DeepLearning/mnt/Devin/boundary/USA_State.json",
):
    """
    description: plot the US map with different data fusion results
    @param {string} fusion_output_file: the data fusion output file
    @param {string} model_file: the model file used for data fusion
    @param {string} base_save_path: the base path to save the plot
    @param {string} boundary_json_file: the boundary file for the US map
    @return None
    """
    mp = model_attribute(model_file)
    proj, longitudes, latitudes = mp.projection, mp.lons, mp.lats
    df_data = pd.read_csv(fusion_output_file)
    if "Period" not in df_data.columns:
        print("The data fusion file does not contain the Period column!")
        return

    # 提取年份
    year = get_year(fusion_output_file)
    # 生成轴标签
    label = get_axis_label(fusion_output_file, year=year)

    # 动态生成路径名称
    if base_save_path:
        save_path = os.path.join(base_save_path, f"Map_{year}_{label}")
        # 如果路径不存在，则自动创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")
        else:
            print(f"Directory already exists: {save_path}")
    else:
        save_path = None

    layout = None
    periods = df_data["Period"].unique()
    for period in tqdm(periods):
        dict_data = {}
        df_period = df_data[df_data["Period"] == period]
        grid_concentration_vna_ozone = df_period["vna_ozone"].values.reshape(
            longitudes.shape
        )
        grid_concentration_evna_ozone = df_period["evna_ozone"].values.reshape(
            longitudes.shape
        )
        key_period = extract_key_period(period)
        if key_period:
            period_label = f"{label}_{key_period}"
        else:
            period_label = label
        period = (
            period.replace("_daily_DF", "")
            .replace("average", "Avg.")
            .replace("Average", "Avg.")
        )
        vmax_conc = max(
            np.nanpercentile(grid_concentration_vna_ozone, 99.5),
            np.nanpercentile(grid_concentration_evna_ozone, 99.5),
        )
        vmin_conc = min(
            np.nanpercentile(grid_concentration_vna_ozone, 0.5),
            np.nanpercentile(grid_concentration_evna_ozone, 0.5),
        )
        value_range = (vmin_conc, vmax_conc)

        get_multiple_data(
            dict_data,
            dataset_name=f"{period}_vna_ozone",
            variable_name="",
            grid_x=longitudes,
            grid_y=latitudes,
            grid_concentration=grid_concentration_vna_ozone,
        )

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
            show_domain_mean=True,  # 用于非全域剔除
        )

        if save_path is not None:
            save_file = os.path.join(save_path, f"{period_label}_vna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period_label} is saved to {save_file}")
        dict_data = {}

        get_multiple_data(
            dict_data,
            dataset_name=f"{period}_evna_ozone",
            variable_name="",
            grid_x=longitudes,
            grid_y=latitudes,
            grid_concentration=grid_concentration_evna_ozone,
        )

        layout = (3, 4) if len(dict_data) == 12 else None
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
            save_file = os.path.join(save_path, f"{period_label}_evna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period_label} is saved to {save_file}")


if __name__ == "__main__":
    base_save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    file_list = [
        "/DeepLearning/mnt/shixiansheng/data_fusion/output/DFT_VNAeVNA_2011_dailyIndex.csv"
    ]
    for file in file_list:
        plot_us_map(file, model_file, base_save_path)
    print("Done!")