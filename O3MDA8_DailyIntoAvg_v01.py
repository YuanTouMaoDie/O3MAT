import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm


def top_10_average(series):
    return series.nlargest(10).mean()


def calculate_metrics_for_grid(grid_df):
    metrics = []

    # 保留 ROW 和 COL 列信息
    row_col = grid_df[["ROW", "COL"]].iloc[0]

    # top-10 average of MDA8 ozone days
    df_data_top_10_avg = grid_df.agg(
        {'vna_ozone': top_10_average,
         'evna_ozone': top_10_average,
         'avna_ozone': top_10_average,
         'model': top_10_average}
    ).to_frame().T
    df_data_top_10_avg["Period"] = f"top-10"
    # 将 ROW 和 COL 信息添加到结果中
    df_data_top_10_avg["ROW"] = row_col["ROW"]
    df_data_top_10_avg["COL"] = row_col["COL"]
    metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = grid_df.groupby(['Year']).agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    # 将 ROW 和 COL 信息添加到结果中
    df_data_annual_avg["ROW"] = row_col["ROW"]
    df_data_annual_avg["COL"] = row_col["COL"]
    metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = grid_df[grid_df['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
    ).to_frame().T
    df_data_summer_avg["Period"] = f"Apr-Sep"
    # 将 ROW 和 COL 信息添加到结果中
    df_data_summer_avg["ROW"] = row_col["ROW"]
    df_data_summer_avg["COL"] = row_col["COL"]
    metrics.append(df_data_summer_avg)

    # seasonal averages（DJF, MAM, JJA, SON）of MDA8
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    for season, months in seasons.items():
        df_data_season = grid_df[grid_df['Month'].isin(months)]
        df_data_season_avg = df_data_season.agg(
            {'vna_ozone': 'mean',
             'evna_ozone': 'mean',
             'avna_ozone': 'mean',
             'model': 'mean'}
        ).to_frame().T
        df_data_season_avg["Period"] = f"{season}"
        # 将 ROW 和 COL 信息添加到结果中
        df_data_season_avg["ROW"] = row_col["ROW"]
        df_data_season_avg["COL"] = row_col["COL"]
        metrics.append(df_data_season_avg)

    final_grid_metrics = pd.concat(metrics, ignore_index=True)
    return final_grid_metrics


def save_daily_data_fusion_to_metrics(df_data, save_path, project_name):
    output_file_list = []

    # 从 Date 列提取年份和月份
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data['Year'] = df_data['Date'].dt.year
    df_data['Month'] = df_data['Date'].dt.month

    # 只保留2011年的数据
    df_data = df_data[df_data['Year'] == 2011]

    all_metrics = []
    grid_groups = list(df_data.groupby(["ROW", "COL"]))

    # 按网格分组并行计算，并显示进度条
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for _, grid_df in grid_groups:
            future = executor.submit(calculate_metrics_for_grid, grid_df)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing grids"):
            result = future.result()
            all_metrics.append(result)

    # 合并所有网格的指标
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}")
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)
    return output_file_list


def process_csv(input_file_path, save_path, project_name):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(input_file_path)
        # 处理数据并保存指标文件
        output_files = save_daily_data_fusion_to_metrics(df, save_path, project_name)
        print(f"处理完成，指标文件已保存到: {output_files}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {input_file_path}")
    except Exception as e:
        print(f"错误: 发生未知错误 {e}")


if __name__ == "__main__":
    input_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_O3MDA8_HourlyIntoDaily.csv'  # 替换为实际的输入 CSV 文件路径
    save_path = '.'  # 替换为实际的保存路径
    project_name = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_Data_WithoutCV/2011_O3MDA8_HourlyIntoDailyIntoMetrics.csv'  # 替换为实际的项目名称
    process_csv(input_file, save_path, project_name)