import pandas as pd

def filter_points_in_region(df, min_lat, max_lat, min_lon, max_lon):
    """
    Filter out points within the specified geographical region.
    @param df: DataFrame containing monitoring data with Lat/Lon columns.
    @param min_lat: Minimum latitude for the region.
    @param max_lat: Maximum latitude for the region.
    @param min_lon: Minimum longitude for the region.
    @param max_lon: Maximum longitude for the region.
    @return: DataFrame with points outside the specified region.
    """
    # Filter points based on latitude and longitude
    df_filtered = df[~((df['Lat'] >= min_lat) & (df['Lat'] <= max_lat) &
                       (df['Lon'] >= min_lon) & (df['Lon'] <= max_lon))]
    
    # Return the filtered dataframe
    return df_filtered

# Example usage
if __name__ == "__main__":
    # Path to the input data
    data_fusion_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    
    # Read the monitoring data
    df_data = pd.read_csv(data_fusion_file)

    # Define the region boundaries
    min_lat = 28.7  # Minimum latitude (28.7N)
    max_lat = 32.7  # Maximum latitude (32.7N)
    min_lon = -118.4  # Minimum longitude (118.4W)
    max_lon = -114.4  # Maximum longitude (114.4W)

    # Filter the data by removing points within the specified region
    df_filtered = filter_points_in_region(df_data, min_lat, max_lat, min_lon, max_lon)

    # Path to save the filtered data
    save_filtered_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/ds.input.aqs.o3.2011_FilterLatLon.csv"
    
    # Save the filtered data to a new file
    df_filtered.to_csv(save_filtered_file, index=False)

    print("Filtering complete. Filtered data saved to:", save_filtered_file)
