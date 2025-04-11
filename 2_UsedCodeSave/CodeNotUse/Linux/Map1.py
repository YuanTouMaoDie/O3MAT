#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


def extract_key_period(period):
    key_periods = ["DJF", "MAM", "JJA", "SON", 'Annual', 'Apr-Sep', 'top-10','W126']
    for key in key_periods:
        if key in period:
            return key
    return None


def get_year(filename):
    match = re.search(r"(20[1-2][0-9])", filename)
    if match:
        return match.group(1)
    return None


def get_dataset_label(variable, filename):
    if variable == 'vna_ozone':
        return "VNA"
    elif variable == 'evna_ozone':
        return "eVNA"
    elif variable == 'avna_ozone':
        return "aVNA"
    elif variable == 'ds_ozone':
        return "Downscaler"
    elif variable =='model':
        return "EQUATES"
    elif variable == 'harvard_ml':
        return "Harvard ML"
    elif variable == 'SD':
        return "Standard Deviation"
    elif variable == 'CV':
        return "Coefficient of Variation"
    return "unknown"


def plot_us_map(
        fusion_output_file,
        model_file,
        base_save_path=None,
        boundary_json_file="/DeepLearning/mnt/Devin/boundary/USA_State.json",
        variable_settings=None,
        key_periods=None
):
    mp = model_attribute(model_file)
    proj, longitudes, latitudes = mp.projection, mp.lons, mp.lats
    df_data = pd.read_csv(fusion_output_file)
    if "Period" not in df_data.columns:
        print("The data fusion file does not contain the Period column!")
        return

    year = get_year(fusion_output_file)
    if not year:
        print("Could not extract year from the filename.")
        return

    save_path = os.path.join("/DeepLearning/mnt/shixiansheng/data_fusion/output", f"{year}_AloneMap_4Hours")
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")
        else:
            print(f"Directory already exists: {save_path}")
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    layout = None
    periods = df_data["Period"].unique()

    for period in tqdm(periods):
        key_period = extract_key_period(period)
        if key_periods and key_period not in key_periods:
            continue

        df_period = df_data[df_data["Period"] == period]
        period_label = f"{year}_{key_period}" if key_period else f"{year}"
        period = (
            period.replace("_daily_DF", "")
           .replace("average", "Avg.")
           .replace("Average", "Avg.")
        )

        for variable in variable_settings['variables']:
            if variable not in df_period.columns:
                print(f"Variable {variable} not found in data for period {period}. Skipping...")
                continue

            grid_concentration = df_period[variable].values.reshape(longitudes.shape)

            value_range = variable_settings['settings'].get('value_range')

            if value_range is None:
                vmax_conc = np.nanpercentile(grid_concentration, 99.5)
                vmin_conc = np.nanpercentile(grid_concentration, 0.5)
                value_range = (vmin_conc, vmax_conc)

            dataset_label = get_dataset_label(variable, fusion_output_file)
            title = f"{period_label}: {dataset_label}"

            dict_data = {}
            get_multiple_data(
                dict_data,
                dataset_name=title,
                variable_name="",
                grid_x=longitudes,
                grid_y=latitudes,
                grid_concentration=grid_concentration
            )

            fig = show_maps(
                dict_data,
                unit=variable_settings['settings'].get('unit', "ppbv"),
                cmap=variable_settings['settings'].get('cmap_conc', cmap_conc),
                show_lonlat=variable_settings['settings'].get('show_lonlat', False),
                projection=proj,
                is_wrf_out_data=variable_settings['settings'].get('is_wrf_out_data', True),
                boundary_file=boundary_json_file,
                show_original_grid=variable_settings['settings'].get('show_original_grid', True),
                panel_layout=variable_settings['settings'].get('panel_layout', layout),
                title_fontsize=variable_settings['settings'].get('title_fontsize', 11),
                xy_title_fontsize=variable_settings['settings'].get('xy_title_fontsize', 9),
                show_dependenct_colorbar=variable_settings['settings'].get('show_dependenct_colorbar', True),
                value_range=value_range,
                show_domain_mean=variable_settings['settings'].get('show_domain_mean', True),
                show_grid_line = variable_settings['settings'].get('show_grid_line', True)
            )

            if save_path is not None:
                save_file = os.path.join(save_path, f"{title}.png")
                fig.savefig(save_file, dpi=300)
                print(f"The data fusion plot for {title} is saved to {save_file}")


if __name__ == "__main__":
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    file_list = ["/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_W126_4HoursF.csv"]

    #special metrics
    # key_periods = ['DJF']
    key_periods = ['W126']
    # key_periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep', 'top-10']

    common_settings = {
        'unit': "ppbv",
        'cmap_conc': cmaps.WhiteBlueGreenYellowRed,
       'show_lonlat': False,
        'is_wrf_out_data': True,
       'show_original_grid': True,
        'panel_layout': None,
        'title_fontsize': 11,
        'xy_title_fontsize': 9,
       'show_dependenct_colorbar': True,
       'show_domain_mean': True,
       'show_grid_line': None,
        'value_range': (0,20),
        # 'value_range': (None,None),
    }

    variable_settings = {
        'variables': ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml'],
        'settings': common_settings
     }
    
    #自己设置
    #variable_settings = {
    #    'variables': ['CV'],
    #   'settings': common_settings
    #}

    for file in file_list:
        plot_us_map(file, model_file, variable_settings=variable_settings, key_periods=key_periods)
    print("Done!")


# In[ ]:




