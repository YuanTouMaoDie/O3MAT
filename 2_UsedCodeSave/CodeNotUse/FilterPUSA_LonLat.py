import os
import pandas as pd
from shapely.geometry import Point, Polygon
from esil.rsm_helper.model_property import model_attribute

#经纬度判断出现偏移

def generate_polygon_within_usa():
    """
    定义美国的多边形区域
    返回：Polygon 对象
    """
    #From DataFusionTool
    coords = [
        (-121.5367, 47.3921),
        (-112.0372, 46.6568),
        (-90.6937, 46.6298),
        (-83.0164, 41.6843),
        (-74.0386, 41.5823),
        (-83.1890, 31.4584),
        (-97.4922, 32.7920),
        (-109.9439, 33.0041),
        (-120.3418, 37.1646)
    ]
    return Polygon(coords)

# def read_txt(file_path):
#     with open(file_path, "r") as f:
#         coords = [tuple(map(float, line.strip().split(", "))) for line in f]
#     return coords

# # 使用方法
# # 该txt文本中为美国本土地区的坐标信息，经由json文件中提取而得
# def generate_polygon_within_usa():
#     coords = read_txt("/DeepLearning/mnt/shixiansheng/data_fusion/USA_Corrected_Boundary_Coordinates.txt")
#     return Polygon(coords)

def add_coordinates_from_model(df, model_file):
    """
    根据 model_file 提供的投影信息，将 ROW 和 COL 转换为经纬度
    @param df: 原始数据（包含 ROW 和 COL）
    @param model_file: NetCDF 模型文件
    @return: 添加经纬度后的数据
    """
    # 解析模型文件，获取投影信息
    mp = model_attribute(model_file)
    longitudes, latitudes = mp.lons, mp.lats  # 获取经纬度数组

    # 确保 ROW 和 COL 在有效范围内
    max_row, max_col = latitudes.shape  # 获取最大行列数
    df = df[(df["ROW"] >= 0) & (df["ROW"] <= max_row) & (df["COL"] >= 0) & (df["COL"] <= max_col)]

    # 计算经纬度
    df["longitude"] = df["COL"].apply(lambda col: longitudes[0, col-1])
    df["latitude"] = df["ROW"].apply(lambda row: latitudes[row-1, 0])

    return df

def filter_points_inside_polygon(df, save_filtered_file, remove_outside=True, fixed_value=0):
    """
    过滤掉不在自定义多边形内的数据，或者将数据的指定列替换为固定值
    @param df: 数据表，已包含经纬度
    @param save_filtered_file: 过滤后数据的保存路径
    @param remove_outside: 是否删除不在多边形内的数据，默认为 True（删除）
    @param fixed_value: 如果 remove_outside=False, 设置替换的固定值，默认为0
    """
    usa_polygon = generate_polygon_within_usa()
    
    # 记录原始数据行数
    original_row_count = len(df)

    if remove_outside:
        # 过滤点
        df_filtered = df[df.apply(lambda row: usa_polygon.contains(Point(row["longitude"], row["latitude"])), axis=1)]
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
        outside_condition = ~df.apply(lambda row: usa_polygon.contains(Point(row["longitude"], row["latitude"])), axis=1)
        df_copy.loc[outside_condition, ['vna_ozone', 'evna_ozone', 'model']] = fixed_value

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
    data_fusion_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_data_fusion_DFT_FtA.csv"
    filtered_file = os.path.join(save_path, "2011_FtA_DFT_filteredForMap.csv")

    # 读取数据
    df_data = pd.read_csv(data_fusion_file)

    # 添加经纬度
    df_with_coords = add_coordinates_from_model(df_data, model_file)

    # 选择是否剔除，或替换为固定值
    remove_outside = False  # 设置为 True 则剔除，设置为 False 则替换为固定值
    fixed_value = 2         # 如果选择替换为固定值，则设置为0（可以根据需求更改）

    # 过滤并保存数据
    filter_points_inside_polygon(df_with_coords, filtered_file, remove_outside=remove_outside, fixed_value=fixed_value)

    print("Done!")
