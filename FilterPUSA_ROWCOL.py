import os
import pandas as pd
from shapely.geometry import Point, Polygon
import pyproj
import numpy as np
import json

def read_json(file_path):
    """读取JSON文件并返回包含坐标的列表"""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    coordinates_list = []
    if "coordinates" in data:
        coordinates_list.append(data["coordinates"])
    elif "features" in data:  # 处理GeoJSON格式
        for feature in data["features"]:
            if "geometry" in feature and feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]:
                geom_type = feature["geometry"]["type"]
                if geom_type == "Polygon":
                    coordinates_list.append(feature["geometry"]["coordinates"][0])  # 取外环坐标
                elif geom_type == "MultiPolygon":
                    for poly in feature["geometry"]["coordinates"]:
                        coordinates_list.append(poly[0])  # 取每个多边形的外环

    return coordinates_list

def generate_polygons_within_usa():
    """定义美国的多个简单多边形区域（从txt文件读取）"""
    coords = read_json("/DeepLearning/mnt/shixiansheng/data_fusion/USA_Contiguous_Boundary_Json.txt")
    
    # 使用 pyproj 将这些经纬度坐标转换为投影坐标（例如 LCC）
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

    # 将每个坐标集合转换为单独的简单多边形
    polygons = []
    for coords_list in coords:
        transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords_list]
        polygons.append(Polygon(transformed_coords))
    
    return polygons

def filter_points_with_multiple_polygons(df, save_filtered_file, polygons, remove_outside=True, fixed_value=0):
    """
    使用多个简单的多边形进行过滤
    @param df: 数据表，已包含经纬度
    @param save_filtered_file: 过滤后数据的保存路径
    @param polygons: 多个Polygon对象的列表
    @param remove_outside: 是否删除不在多边形内的数据，默认为 True（删除）
    @param fixed_value: 如果 remove_outside=False, 设置替换的固定值，默认为0
    """
    # 记录原始数据行数
    original_row_count = len(df)

    # 筛选不在多边形中的点
    def is_point_inside_polygons(row, polygons):
        #lambda指定列中获得COL和ROW并构成点集
        point = Point(row["COL"], row["ROW"])
        return any(poly.contains(point) for poly in polygons)

    if remove_outside:
        # 过滤数据，只保留在任何一个多边形内的点
        df_filtered = df[df.apply(lambda row: is_point_inside_polygons(row, polygons), axis=1)]
        # 记录剔除后的数据行数
        filtered_row_count = len(df_filtered)
        removed_row_count = original_row_count - filtered_row_count

        # 打印相关统计信息
        print(f"原始数据行数: {original_row_count}")
        print(f"剔除数据行数: {removed_row_count}")
        print(f"剩余数据行数: {filtered_row_count}")

        # 保存结果
        df_filtered.to_csv(save_filtered_file, index=False)
    else:
        # 替换数据
        df_copy = df.copy()
        outside_condition = ~df.apply(lambda row: is_point_inside_polygons(row, polygons), axis=1)
        df_copy.loc[outside_condition, ['avna_ozone','model','vna_ozone', 'evna_ozone']] = fixed_value

        # 记录替换后的数据行数
        replaced_row_count = len(df_copy)
        replaced_data_count = outside_condition.sum()

        # 打印相关统计信息
        print(f"原始数据行数: {original_row_count}")
        print(f"替换数据行数: {replaced_data_count}")
        print(f"剩余数据行数: {replaced_row_count}")

        # 保存结果
        df_copy.to_csv(save_filtered_file, index=False)

    print(f"已保存处理后的数据至: {save_filtered_file}")

if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    data_fusion_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/BarronResult_aVNA_2011_FtAIndex.csv"
    filtered_file = os.path.join(save_path, "BarronReuslt_aVNA_2011_FtAIndex_InUSA.csv")

    # 读取数据
    df_data = pd.read_csv(data_fusion_file)

    # 生成美国区域多个简单多边形（保持投影转换）
    usa_polygons = generate_polygons_within_usa()

    # 选择是否剔除，或替换为固定值
    remove_outside = False  # 设置为 True 则剔除，设置为 False 则替换为固定值
    fixed_value = np.NAN         # 如果选择替换为固定值，则设置为0（可以根据需求更改）

    # 过滤并保存数据
    filter_points_with_multiple_polygons(df_data, filtered_file, usa_polygons, remove_outside=remove_outside, fixed_value=fixed_value)

    print("Done!")
