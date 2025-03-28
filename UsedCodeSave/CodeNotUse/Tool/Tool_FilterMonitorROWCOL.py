import os
import pandas as pd
from shapely.geometry import Point, Polygon
import pyproj
import numpy as np
import json

def read_json(file_path):
    """Read the JSON file and return a list of coordinates."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    coordinates_list = []
    if "coordinates" in data:
        coordinates_list.append(data["coordinates"])
    elif "features" in data:  # Handle GeoJSON format
        for feature in data["features"]:
            if "geometry" in feature and feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]:
                geom_type = feature["geometry"]["type"]
                if geom_type == "Polygon":
                    coordinates_list.append(feature["geometry"]["coordinates"][0])  # Get outer ring coordinates
                elif geom_type == "MultiPolygon":
                    for poly in feature["geometry"]["coordinates"]:
                        coordinates_list.append(poly[0])  # Get each polygon's outer ring

    return coordinates_list

def generate_polygons_within_usa():
    """Define multiple simple polygons for the USA (from txt file)."""
    coords = read_json("/DeepLearning/mnt/shixiansheng/data_fusion/USA_Contiguous_Boundary_Json.txt")
    
    # Use pyproj to convert these coordinates into projection coordinates (e.g., LCC)
    proj_string = (
        "+proj=lcc "
        "+lat_0=40 +lon_0=-97 "
        "+lat_1=33 +lat_2=45 "
        "+x_0=2556000 +y_0=1728000 "
        "+R=6370000 "
        "+to_meter=12000 "
        "+no_defs"
    )
    proj = pyproj.CRS.from_proj4(proj_string)
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), proj, always_xy=True)

    # Convert each coordinate list into individual simple polygons
    polygons = []
    for coords_list in coords:
        transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords_list]
        polygons.append(Polygon(transformed_coords))
    
    return polygons

def convert_lat_lon_to_row_col(lat, lon):
    """Convert Lat/Lon to Row/Col coordinates using the same projection as the polygons."""
    # Define the projection string for LCC
    proj_string = (
        "+proj=lcc "
        "+lat_0=40 +lon_0=-97 "
        "+lat_1=33 +lat_2=45 "
        "+x_0=2556000 +y_0=1728000 "
        "+R=6370000 "
        "+to_meter=12000 "
        "+no_defs"
    )
    proj = pyproj.CRS.from_proj4(proj_string)
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), proj, always_xy=True)
    
    # Transform Lat/Lon to Row/Col (x, y)
    row, col = transformer.transform(lon, lat)  # Lon is the first, Lat is the second parameter
    return row, col

def filter_points_by_lat_lon(df, save_filtered_file, polygons, remove_outside=True):
    """
    Use Lat/Lon to filter monitoring data based on polygons.
    Convert Lat/Lon to Row/Col first and then check against polygons.
    @param df: DataFrame containing monitoring data with Lat/Lon.
    @param save_filtered_file: Path to save the filtered data.
    @param polygons: List of Polygon objects.
    @param remove_outside: Whether to remove points outside the polygons. Default is True (remove).
    """
    # Record the original row count
    original_row_count = len(df)

    # Check if a point is inside any of the polygons
    def is_point_inside_polygons(row, polygons):
        # Convert Lat/Lon to Row/Col
        row_coord, col_coord = convert_lat_lon_to_row_col(row["Lat"], row["Lon"])  # Use Lat/Lon directly
        point = Point(col_coord, row_coord)  # Use the transformed coordinates (Row, Col)
        return any(poly.contains(point) for poly in polygons)

    if remove_outside:
        # Filter data, only keep points inside the polygons
        df_filtered = df[df.apply(lambda row: is_point_inside_polygons(row, polygons), axis=1)]
        # Record the filtered data row count
        filtered_row_count = len(df_filtered)
        removed_row_count = original_row_count - filtered_row_count

        # Print relevant statistics
        print(f"Original data row count: {original_row_count}")
        print(f"Rows removed: {removed_row_count}")
        print(f"Remaining data rows: {filtered_row_count}")

        # Save the filtered data
        df_filtered.to_csv(save_filtered_file, index=False)
    else:
        # Replace data for points outside the polygons
        df_copy = df.copy()
        outside_condition = ~df.apply(lambda row: is_point_inside_polygons(row, polygons), axis=1)
        df_copy.loc[outside_condition, ['Conc']] = np.nan  # Replace with NaN or fixed value

        # Record the replaced data row count
        replaced_row_count = len(df_copy)
        replaced_data_count = outside_condition.sum()

        # Print relevant statistics
        print(f"Original data row count: {original_row_count}")
        print(f"Rows replaced: {replaced_data_count}")
        print(f"Remaining data rows: {replaced_row_count}")

        # Save the modified data
        df_copy.to_csv(save_filtered_file, index=False)

    print(f"Processed data has been saved to: {save_filtered_file}")

if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_fusion_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    filtered_file = os.path.join(save_path, "ds.input.aqs.o3.2011_filtered.csv")

    # Read monitoring data
    df_data = pd.read_csv(data_fusion_file)

    # Generate multiple simple polygons for the USA
    usa_polygons = generate_polygons_within_usa()

    # Choose whether to remove points outside or replace them with fixed value
    remove_outside = True  # Set to True to remove, False to replace with fixed value

    # Filter and save data
    filter_points_by_lat_lon(df_data, filtered_file, usa_polygons, remove_outside=remove_outside)

    print("Done!")
