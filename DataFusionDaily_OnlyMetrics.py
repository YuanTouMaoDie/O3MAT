import pandas as pd
import os


def save_daily_data_fusion_to_metrics(df_data, save_path, project_name):
    '''
    This function converts the daily data fusion results to O3-related metrics (e.g., 98th percentile of MDA8 ozone concentration, average of top-10 MDA8 ozone days, annual average of MDA8 ozone concentration) files.
    @param {DataFrame} df_data: The DataFrame of the daily data fusion results.
    @param {str} save_path: The path to save the O3-related metrics files.
    @param {str} project_name: The name of the project.
    @return {list} output_file_list: The list of the O3-related metrics files.
    '''
    output_file_list = []

    # 提取年份和月份
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month

    # 初始化一个空的 DataFrame 来存储所有指标
    all_metrics = []

    # top-10 average of MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()

    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': top_10_average,
         'evna_ozone': top_10_average,
         'avna_ozone': top_10_average,
         'model': top_10_average}
    ).reset_index()
    df_data_top_10_avg["Period"] = f"top-10"
    all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {'vna_ozone': 'mean',
         'evna_ozone': 'mean',
         'avna_ozone': 'mean',
         'model': 'mean'}
    ).reset_index()
    df_data_summer_avg["Period"] = f"Apr-Sep"
    all_metrics.append(df_data_summer_avg)

    # seasonal averages（DJF, MAM, JJA, SON）of MDA8
    seasons = {
        'DJF': [12, 1, 2],  # December,January, Feburary
        'MAM': [3, 4, 5],  # April, May, June
        'JJA': [6, 7, 8],  # July, August, September
        'SON': [9, 10, 11]  # October, November, December
    }
    for season, months in seasons.items():
        df_data_season = df_data[df_data['Month'].isin(months)]
        df_data_season_avg = df_data_season.groupby(["ROW", "COL"]).agg(
            {'vna_ozone': 'mean',
             'evna_ozone': 'mean',
             'avna_ozone': 'mean',
             'model': 'mean'}
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}_Metrics.csv")
    final_df.to_csv(output_file, index=False)
    output_file_list.append(output_file)

    return output_file_list


if __name__ == "__main__":
    base_save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/"

    year = 2011
    # 假设已经有了处理好的每日融合数据文件
    file_path = os.path.join(base_save_path, f"{year}_Data_WithoutCV.csv")
    df_data = pd.read_csv(file_path)

    save_path = os.path.dirname(file_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    project_name = os.path.basename(file_path).replace(".csv", "")
    output_file_list = save_daily_data_fusion_to_metrics(df_data, save_path, project_name)
    print(f"O3-related metrics files saved for year {year}: {output_file_list}")

    