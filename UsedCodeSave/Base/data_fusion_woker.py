import pyrsig
import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
#for show maps
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data,show_maps
import cmaps
cmap_conc=cmaps.WhiteBlueGreenYellowRed   
cmap_delta = cmaps.ViBlGrWhYeOrRe

@timer_decorator
def start_daily_data_fusion(
    model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3"
):
    """
    @param {string} model_file, the model file must have dimensions: Time,Layer, ROW, COL, and variables: O3_MDA8
    @param {string} monitor_file, the monitor file must have columns: Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: output file path
    @param {string} monitor_pollutant: the pollutant in monitor file, default is ozone
    @param {string} model_pollutant: the pollutant in model file, default is O3
    """
    # Fit/predict model for ozone, model ozone, and bias
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    # Group the observations by site and date, and calculate the mean concentration, latitude, and longitude for each group, and then order the data by date
    # df_obs_grouped = df_obs.groupby(["Site", "Date"]).aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"}).reset_index().sort_values(by="Date")
    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
        .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
        .sort_values(by="Date")
    ).reset_index()
    dates = df_obs_grouped["Date"].unique()
    with tqdm(dates) as pbar:
        for date in pbar:
            pbar.set_description(f"Data Fusion for {date}...")
            # Record the start time
            start_time = time.time()
            df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()            
            # Get the model data for the current date, only have TSTEP rather than Time, so we need to select the data by date
            if isinstance(ds_model['TSTEP'].values[0],np.int64):
                timeIndex = get_day_of_year(date) - 1
                ds_daily_model = ds_model.sel(TSTEP=timeIndex)
            else:
                ds_daily_model = ds_model.sel(TSTEP=date)
            df_daily_obs["x"], df_daily_obs["y"] = proj(
                df_daily_obs["Lon"], df_daily_obs["Lat"]
            )
            df_daily_obs["mod"] = ds_daily_model[model_pollutant][0].sel(
                ROW=df_daily_obs["y"].to_xarray(),
                COL=df_daily_obs["x"].to_xarray(),
                method="nearest",
            )
            df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
            # Calculate r_n, added by Devin
            df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

            df_prediction = ds_daily_model[["ROW", "COL"]].to_dataframe().reset_index() #TODO: check if this is correct
            nn.fit(
                df_daily_obs[["x", "y"]],
                df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]],
            )  # modified by Devin

            zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
            # Add results to prediction dataframe
            (
                df_prediction["vna_ozone"],
                df_prediction["vna_mod"],
                df_prediction["vna_bias"],
                df_prediction["vna_r_n"],
            ) = zdf.T  # modified by Devin
            # Convert dataframe to netcdf-like object
            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()            
            df_fusion["avna_ozone"] = (
                ds_daily_model[model_pollutant][0].values - df_fusion["vna_bias"]
            )
            # Reshape the vna_r_n data to match the shape of the model data
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(
                ds_daily_model[model_pollutant][0].shape
            )
            # Set the evna_ozone variable with explicit dimension names
            df_fusion["evna_ozone"] = (
                ("ROW", "COL"),
                ds_daily_model[model_pollutant][0].values * reshaped_vna_r_n,
            )
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"]=ds_daily_model[model_pollutant][0].values.flatten()
            df_fusion["Timestamp"] = date
            # convert row and col from float to int, and add 0.5 to avoid 0-based indexing
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat(
                    [df_all_daily_prediction, df_fusion]
                )
            # Record the end time
            end_time = time.time()
            # Calculate the duration
            duration = end_time - start_time
            # Print or log the duration
            print(f"Data Fusion for {date} took {duration:.2f} seconds")
        df_all_daily_prediction.to_csv(file_path, index=False)
        project_name=os.path.basename(file_path).replace(".csv","")        
        save_daily_data_fusion_to_metrics(file_path,save_path,project_name)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")


def start_period_averaged_data_fusion(
    model_file,
    monitor_file,
    file_path,
    dict_period,
    monitor_pollutant="ozone",
    model_pollutant="O3",
):
    """
    @param {string} model_file, the model file must have dimensions: Time,Layer, ROW, COL, and variables: O3_MDA8
    @param {string} monitor_file, the monitor file must have columns: Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: output file path
    @param {dict} dict_period: the period of each data fusion, the key is the period name, and the value is a list of start and end date of the period. 
    e.g., {"JFM_2011": ["2011-01-01", "2011-03-31"], "AMJ_2011": ["2011-04-01", "2011-06-30"], "JAS_2011": ["2011-07-01", "2011-09-30"], "OND_2011": ["2011-10-01", "2011-12-31"]}
    """
    # Fit/predict model for ozone, model ozone, and bias
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    # Group the observations by site and date, and calculate the mean concentration, latitude, and longitude for each group, and then order the data by date
    # df_obs_grouped = df_obs.groupby(["Site", "Date"]).aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"}).reset_index().sort_values(by="Date")
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])

    with tqdm(dict_period.items()) as pbar:
        for peroid_name, peroid in pbar:
            start_date, end_date = peroid[0], peroid[1]
            pbar.set_description(f"Data Fusion for {peroid_name}...")

            if start_date > end_date:  # 处理跨越时间段
                df_filtered_obs = df_obs[
                    (df_obs["Date"] >= start_date) | (df_obs["Date"] <= end_date)
                ]
            else:
                df_filtered_obs = df_obs[
                    (df_obs["Date"] >= start_date) & (df_obs["Date"] <= end_date)
                ]
            df_avg_obs = (
                df_filtered_obs.groupby(["Site"]).aggregate(
                    {"Conc": "mean", "Lat": "mean", "Lon": "mean"}
                )

            ).reset_index()
            # Record the start time
            start_time = time.time()
            # df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
            # Get the model data for the current date, only have TSTEP rather than Time, so we need to select the data by date
            
            # if isinstance(ds_model["TSTEP"].values[0], np.int64):
            #     start_time_index = get_day_of_year(start_date) - 1
            #     end_time_index = get_day_of_year(end_date) - 1
            #     ds_avg_model = ds_model.sel(
            #         TSTEP=slice(start_time_index, end_time_index)
            #     ).mean(dim="TSTEP")
            # else:
            #     ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(
            #         dim="TSTEP"
            #     )
            if isinstance(ds_model["TSTEP"].values[0], np.int64):
                start_time_index = get_day_of_year(start_date) - 1
                end_time_index = get_day_of_year(end_date) - 1
                if start_date > end_date:  # 跨年情况
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(range(start_time_index, len(ds_model["TSTEP"]))) + list(range(0, end_time_index + 1))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(
                        TSTEP=slice(start_time_index, end_time_index)
                    ).mean(dim="TSTEP")
            else:
                if start_date > end_date:  # 跨年情况
                    ds_avg_model = ds_model.sel(
                        TSTEP=list(pd.date_range(start=start_date, end='2011-12-31')) + list(pd.date_range(start='2011-01-01', end=end_date))
                    ).mean(dim="TSTEP")
                else:
                    ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(
                        dim="TSTEP"
                    )
            
            df_avg_obs["x"], df_avg_obs["y"] = proj(
                df_avg_obs["Lon"], df_avg_obs["Lat"]
            )
            df_avg_obs["mod"] = ds_avg_model[model_pollutant][0].sel(
                ROW=df_avg_obs["y"].to_xarray(),
                COL=df_avg_obs["x"].to_xarray(),
                method="nearest",
            )
            df_avg_obs["bias"] = df_avg_obs["mod"] - df_avg_obs["Conc"]
            # Calculate r_n, added by Devin
            df_avg_obs["r_n"] = df_avg_obs["Conc"] / df_avg_obs["mod"]

            df_prediction = ds_avg_model[["ROW", "COL"]].to_dataframe().reset_index()
            nn.fit(
                df_avg_obs[["x", "y"]],
                df_avg_obs[[monitor_pollutant, "mod", "bias", "r_n"]],
            )  # modified by Devin

            zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
            # Add results to prediction dataframe
            (
                df_prediction["vna_ozone"],
                df_prediction["vna_mod"],
                df_prediction["vna_bias"],
                df_prediction["vna_r_n"],
            ) = zdf.T  # modified by Devin
            # Convert dataframe to netcdf-like object
            df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()            
            df_fusion["avna_ozone"] = (
                ds_avg_model[model_pollutant][0].values - df_fusion["vna_bias"]
            )            
            # Reshape the vna_r_n data to match the shape of the model data
            reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(
                ds_avg_model[model_pollutant][0].shape
            )
            # Set the evna_ozone variable with explicit dimension names
            df_fusion["evna_ozone"] = (
                ("ROW", "COL"),
                ds_avg_model[model_pollutant][0].values * reshaped_vna_r_n,
            )
            df_fusion = df_fusion.to_dataframe().reset_index()
            df_fusion["model"] = ds_avg_model[model_pollutant][0].values.flatten()
            df_fusion["Period"] = peroid_name
            # convert row and col from float to int, and add 0.5 to avoid 0-based indexing
            df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
            df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
            if df_all_daily_prediction is None:
                df_all_daily_prediction = df_fusion
            else:
                df_all_daily_prediction = pd.concat(
                    [df_all_daily_prediction, df_fusion]
                )
            # Record the end time
            end_time = time.time()
            # Calculate the duration
            duration = end_time - start_time
            # Print or log the duration
            print(f"Data Fusion for {peroid_name} took {duration:.2f} seconds")
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(
            f"Data Fusion for all dates is done, the results are saved to {file_path}"
        )
  

@timer_decorator
def save_daily_data_fusion_to_metrics(daily_data_fusion_file,save_path,project_name):
    '''
    This function converts the daily data fusion results to O3-related metrics (e.g., 98th percentile of MDA8 ozone concentration, average of top-10 MDA8 ozone days, annual average of MDA8 ozone concentration) files.
    @param {str} daily_data_fusion_file: The path of the daily data fusion results file.
    @param {str} save_path: The path to save the O3-related metrics files.
    @param {str} project_name: The name of the project.
    @return {list} output_file_list: The list of the O3-related metrics files.
    '''
    output_file_list =[]
    df_data = pd.read_csv(daily_data_fusion_file)
    #region 98th percentile of MDA8 ozone concentration
    # Group by COL and ROW, then calculate the 98th percentile for each specified column
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data_98th_percentile = df_data.groupby(["ROW", "COL"]).agg(
        {'model': lambda x: x.quantile(0.98),
        'vna_ozone': lambda x: x.quantile(0.98),
        'evna_ozone': lambda x: x.quantile(0.98),
        'avna_ozone': lambda x: x.quantile(0.98),
        }
    ).reset_index()
    period_name=f"{project_name}_98th percentile O3-MDA8"
    df_data_98th_percentile["Period"]=period_name
    df_data_98th_percentile.to_csv(os.path.join(save_path, f"{period_name}.csv"), index=False)
    output_file_list.append(os.path.join(save_path, f"{period_name}.csv"))
    #endregion 98th percentile of MDA8 ozone concentration
    
    #region top-10 average of MDA8 ozone days
    # Function to calculate the average of the top-10 MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()
    # Group by COL and ROW, then calculate the average of the top-10 MDA8 ozone days for each specified column
    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'model': top_10_average,
        'vna_ozone': top_10_average,
        'evna_ozone': top_10_average,
        'avna_ozone': top_10_average}
    ).reset_index()
    period_name=f"{project_name}_Average of top-10 O3-MDA8 days"
    df_data_top_10_avg["Period"]=period_name
    df_data_top_10_avg.to_csv(os.path.join(save_path, f"{period_name}.csv"), index=False)
    output_file_list.append(os.path.join(save_path, f"{period_name}.csv"))
    #endregion top-10 average of MDA8 ozone days
    
    #region Annual average of MDA8
    # Extract the year from the Timestamp column
    df_data['Year'] = df_data['Timestamp'].dt.year
    # Group by COL, ROW, and Year, then calculate the mean of MDA8 ozone concentration for each specified column
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'model': 'mean',
        'vna_ozone': 'mean',
        'evna_ozone': 'mean',
        'avna_ozone': 'mean'}
    ).reset_index()
    period_name=f"{project_name}_Annual O3-MDA8"
    df_data_annual_avg["Period"]=period_name
    df_data_annual_avg.to_csv(os.path.join(save_path, f"{period_name}.csv"), index=False)    
    output_file_list.append(os.path.join(save_path, f"{period_name}.csv"))  
    #endregion Annual average of MDA8

    #region Summer season average (Apr-Sep) of MDA8
    # Extract the month from the Timestamp column
    df_data['Month'] = df_data['Timestamp'].dt.month

    # Filter the data to include only the summer months (Apr-Sep)
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    # Group by COL and ROW, then calculate the mean of MDA8 ozone concentration for each specified column in the summer months
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'model': 'mean',
        'vna_ozone': 'mean',
        'evna_ozone': 'mean',
        'avna_ozone': 'mean'}
    ).reset_index()
    period_name=f"{project_name}_Summer season average (Apr-Sep) of O3-MDA8"
    df_data_summer_avg["Period"]=period_name
    df_data_summer_avg.to_csv(os.path.join(save_path, f"{period_name}.csv"), index=False)
    output_file_list.append(os.path.join(save_path, f"{period_name}.csv"))
    #endregion Summer season average (Apr-Sep) of MDA8
    
    #region seasonal averages（JFM, AMJ, JAS, OND）of MDA8
    
    # Define the seasons
    # seasons = {
    #     'JFM': [1, 2, 3],  # January, February, March
    #     'AMJ': [4, 5, 6],  # April, May, June
    #     'JAS': [7, 8, 9],  # July, August, September
    #     'OND': [10, 11, 12]  # October, November, December
    # }
    seasons = {
         'DJF': [12, 1, 2],  # December,January, Feburary
         'MAM': [3, 4, 5],  # April, May, June
         'JJA': [6, 7, 8],  # July, August, September
         'SON': [9, 10, 11]  # October, November, December
    }
    # seasonal_averages = {}
    df_final_seasonal_avg = None
    for season, months in seasons.items():
        df_data_season = df_data[df_data['Month'].isin(months)]
        df_data_season_avg = df_data_season.groupby(["ROW", "COL"]).agg(
            {'model': 'mean',
            'vna_ozone': 'mean',
            'evna_ozone': 'mean',
            'avna_ozone': 'mean'}
        ).reset_index()
        df_data_season_avg["Period"]=f"{project_name}_{season}_O3-MDA8"
        if df_final_seasonal_avg is None:
            df_final_seasonal_avg = df_data_season_avg
        else:
            df_final_seasonal_avg = pd.concat([df_final_seasonal_avg, df_data_season_avg], ignore_index=True)
    df_final_seasonal_avg.to_csv(os.path.join(save_path, f"{project_name}_seasonal_O3-MDA8.csv"), index=False)
    output_file_list.append(os.path.join(save_path, f"{project_name}_seasonal_O3-MDA8.csv"))
    #endregion Seasonal averages（JFM, AMJ, JAS, OND）of MDA8
        
    return output_file_list
      
def plot_us_map(fusion_output_file,model_file,save_path=None,boundary_json_file="/DeepLearning/mnt/Devin/boundary/USA_State.json"):
    '''
    description: plot the US map with different data fusion results
    @param {string} fusion_output_file: the data fusion output file
    @param {string} model_file: the model file used for data fusion
    @param {string} save_path: the path to save the plot
    @param {string} boundary_json_file: the boundary file for the US map
    @return None
    '''
    mp = model_attribute(model_file)
    proj, longitudes, latitudes = mp.projection,mp.lons, mp.lats
    df_data = pd.read_csv(fusion_output_file)
    if 'Period' not in df_data.columns:
        print("The data fusion file does not contain the Period column!")
        return
    layout= None    
    periods=df_data["Period"].unique()                
    for period in tqdm(periods):
        dict_data={}
        df_period=df_data[df_data["Period"]==period]
        grid_concentration_model=df_period["model"].values.reshape(longitudes.shape)
        grid_concentration_vna_ozone=df_period["vna_ozone"].values.reshape(longitudes.shape)
        # grid_concentration_vna_mod=df_period["vna_mod"].values.reshape(longitudes.shape)
        grid_concentration_avna_ozone=df_period["avna_ozone"].values.reshape(longitudes.shape)
        grid_concentration_evna_ozone=df_period["evna_ozone"].values.reshape(longitudes.shape)
        period=period.replace("_daily_DF","").replace("average","Avg.").replace("Average","Avg.")
        vmax_conc = max(np.nanpercentile(grid_concentration_model, 99.5),np.nanpercentile(grid_concentration_vna_ozone, 99.5),np.nanpercentile(grid_concentration_avna_ozone, 99.5),np.nanpercentile(grid_concentration_evna_ozone, 99.5))
        vmin_conc = min(np.nanpercentile(grid_concentration_model, 0.5),np.nanpercentile(grid_concentration_vna_ozone, 0.5),np.nanpercentile(grid_concentration_avna_ozone, 0.5),np.nanpercentile(grid_concentration_evna_ozone, 0.5))
        value_range=(vmin_conc, vmax_conc)
        get_multiple_data(dict_data,dataset_name=f"{period}_EQUATES",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_model) 
        get_multiple_data(dict_data,dataset_name=f"{period}_vna_ozone",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_vna_ozone)
        get_multiple_data(dict_data,dataset_name=f"Delta (vna_ozone - EQUATES)",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_vna_ozone - grid_concentration_model,is_delta=True,cmap=cmap_delta)
        fig=show_maps(dict_data,unit='ppbv',cmap=cmap_conc, show_lonlat=True,projection=proj,is_wrf_out_data=True, boundary_file=boundary_json_file,show_original_grid=True,panel_layout=layout
                        ,delta_map_settings={
                        'cmap':cmap_delta,
                        'value_range':(None,None),
                        'colorbar_ticks_num':None, 
                        'colorbar_ticks_value_format':'.2f',
                        'value_format':'.2f'
                        },title_fontsize=11,xy_title_fontsize=9,show_dependenct_colorbar=True,value_range=value_range)  
        
        if save_path is not None:
            save_file=os.path.join(save_path,f"{period}_vna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period} is saved to {save_file}")
        dict_data={}
        # get_multiple_data(dict_data,dataset_name=f"{period}zz",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_model) 
        # get_multiple_data(dict_data,dataset_name=f"{period}_vna_mod",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_vna_mod)
        # get_multiple_data(dict_data,dataset_name=f"{period} Delta (vna_mod - EQUATES)",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_vna_mod - grid_concentration_model,is_delta=True,cmap=cmap_delta)
        
        get_multiple_data(dict_data,dataset_name=f"{period}_EQUATES",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_model)
        get_multiple_data(dict_data,dataset_name=f"{period}_avna_ozone",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_avna_ozone)
        get_multiple_data(dict_data,dataset_name=f"Delta (avna_ozone - EQUATES)",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_avna_ozone - grid_concentration_model,is_delta=True,cmap=cmap_delta)
        fig=show_maps(dict_data,unit='ppbv',cmap=cmap_conc, show_lonlat=True,projection=proj,is_wrf_out_data=True, boundary_file=boundary_json_file,show_original_grid=True,panel_layout=layout
                      ,delta_map_settings={
                        'cmap':cmap_delta,
                        'value_range':(None,None),
                        'colorbar_ticks_num':None, 
                        'colorbar_ticks_value_format':'.2f',
                        'value_format':'.2f'
                        },title_fontsize=11,xy_title_fontsize=9,show_dependenct_colorbar=True,value_range=value_range)  
        if save_path is not None:
            save_file=os.path.join(save_path,f"{period}_avna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period} is saved to {save_file}")
        dict_data={}    
        get_multiple_data(dict_data,dataset_name=f"{period}_EQUATES",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_model)
        get_multiple_data(dict_data,dataset_name=f"{period}_evna_ozone",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_evna_ozone)
        get_multiple_data(dict_data,dataset_name=f"Delta (evna_ozone - EQUATES)",variable_name="",grid_x=longitudes,grid_y=latitudes,grid_concentration= grid_concentration_evna_ozone - grid_concentration_model,is_delta=True,cmap=cmap_delta)
        
        layout=(3,4) if len(dict_data)==12 else None 
        fig=show_maps(dict_data,unit='ppbv',cmap=cmap_conc, show_lonlat=True,projection=proj,is_wrf_out_data=True, boundary_file=boundary_json_file,show_original_grid=True,panel_layout=layout
                      ,delta_map_settings={
                        'cmap':cmap_delta,
                        'value_range':(None,None),
                        'colorbar_ticks_num':None, 
                        'colorbar_ticks_value_format':'.2f',
                        'value_format':'.2f'
                        },title_fontsize=11,xy_title_fontsize=9,show_dependenct_colorbar=True,value_range=value_range)  
        if save_path is not None:
            save_file=os.path.join(save_path,f"{period}_evna_ozone_data_fusion.png")
            fig.savefig(save_file, dpi=300)
            print(f"The data fusion plot for {period} is saved to {save_file}")
                
if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_file_path = os.path.join(save_path, "daily_fused_data_2011.csv")
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    monitor_file =r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"    
    #测试 先融合后均值
    # start_daily_data_fusion(
    # model_file,
    # monitor_file,
    # output_file_path,
    # monitor_pollutant="Conc",
    # model_pollutant="O3_MDA8",
    # )
    
    # 测试 先均值后融合
    output_file_path = os.path.join(save_path, "seasonal_fused_data_2011.csv")
    start_period_averaged_data_fusion(
        model_file,
        monitor_file,
        output_file_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        # dict_period={
        #     "JFM_2011": ["2011-01-01", "2011-03-31"],
        #      "AMJ_2011": ["2011-04-01", "2011-06-30"],
        #      "JAS_2011": ["2011-07-01", "2011-09-30"],
        #      "OND_2011": ["2011-10-01", "2011-12-31"],
        #     "Annual_2011": ["2011-01-01", "2011-12-31"],
        #     "Apr-Sep_2011": ["2011-01-01", "2011-12-31"]

        # },
          dict_period={
            "DJF_2011": ["2011-12-01", "2011-02-28"],
             "MAM_2011": ["2011-03-01", "2011-05-31"],
             "JJA_2011": ["2011-06-01", "2011-08-31"],
             "SON_2011": ["2011-09-01", "2011-11-30"],
        },
    )
    
    
    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    data_fusion_file =r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011Data_OutputJFM-OND/daily_fused_data_2011.csv"    
    
    #将其处理为相应的指标
    # file_list=save_daily_data_fusion_to_metrics(data_fusion_file,save_path,project_name="2011_daily_DF")
    # file_list=['/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_98th percentile O3-MDA8.csv', '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_Average of top-10 O3-MDA8 days.csv', '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_Annual O3-MDA8.csv', '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_Summer season average (Apr-Sep) of O3-MDA8.csv', '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_seasonal_O3-MDA8.csv']
    file_list=['/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_daily_DF_seasonal_O3-MDA8.csv']
    save_path=os.path.join(save_path,"images")

    #绘图
    # for file in file_list:  
    #     plot_us_map(file,model_file,save_path)
            
    print("Done!")  
    print("Done!")  