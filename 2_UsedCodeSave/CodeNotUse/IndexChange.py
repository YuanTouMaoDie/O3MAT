import pandas as pd
import numpy as np
import os


def save_to_csv(df, output_path):
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")


def convert_data_to_metrics(input_csv_file, save_path, project_name):
    '''
    This function converts the daily data fusion results in a CSV file to O3-related metrics
    (e.g., 98th percentile of MDA8 ozone concentration, average of top-10 MDA8 ozone days,
    annual average of MDA8 ozone concentration) and saves the results in a CSV file.
    @param {str} input_csv_file: The path to the input CSV file containing daily data fusion results.
    @param {str} save_path: The path to save the O3-related metrics CSV file.
    @param {str} project_name: The name of the project.
    @return {str} output_file: The path to the saved O3-related metrics CSV file.
    '''
    # 读取输入的 CSV 文件
    df_data = pd.read_csv(input_csv_file)

    # 提取年份和月份
    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
    df_data['Year'] = df_data['Timestamp'].dt.year
    df_data['Month'] = df_data['Timestamp'].dt.month

    # 初始化一个空的 DataFrame 来存储所有指标
    all_metrics = []

    # 98th percentile of MDA8 ozone concentration
    df_data_98th_percentile = df_data.groupby(["ROW", "COL"]).agg(
        {
         'model': lambda x: x.quantile(0.98),}
    ).reset_index()
    df_data_98th_percentile["Period"] = f"98th"
    all_metrics.append(df_data_98th_percentile)

    # top-10 average of MDA8 ozone days
    def top_10_average(series):
        return series.nlargest(10).mean()

    df_data_top_10_avg = df_data.groupby(["ROW", "COL"]).agg(
        {'model': top_10_average,}
    ).reset_index()
    df_data_top_10_avg["Period"] = f"top-10"
    all_metrics.append(df_data_top_10_avg)

    # Annual average of MDA8
    df_data_annual_avg = df_data.groupby(["ROW", "COL", 'Year']).agg(
        {
         'model': 'mean',
         }
    ).reset_index()
    df_data_annual_avg["Period"] = f"Annual"
    all_metrics.append(df_data_annual_avg)

    # Summer season average (Apr-Sep) of MDA8
    summer_months = [4, 5, 6, 7, 8, 9]
    df_data_summer = df_data[df_data['Month'].isin(summer_months)]
    df_data_summer_avg = df_data_summer.groupby(["ROW", "COL"]).agg(
        {

         'model': 'mean',
}
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
            {
             'model': 'mean',
            }
        ).reset_index()
        df_data_season_avg["Period"] = f"{season}"
        all_metrics.append(df_data_season_avg)

    # 合并所有指标到一个 DataFrame
    final_df = pd.concat(all_metrics, ignore_index=True)

    # 保存为一个 CSV 文件
    output_file = os.path.join(save_path, f"{project_name}_dailyIndex.csv")
    save_to_csv(final_df, output_file)

    return output_file


if __name__ == "__main__":
    # 输入 CSV 文件路径
    input_csv_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/EQUATES_model_2011_daily.csv'  # 请替换为实际路径

    # 保存路径
    save_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output'  # 请替换为实际路径

    # 项目名称
    project_name = 'EQUATES_model_2011'  # 输出数据表的名称

    # 转换数据并保存
    output_file = convert_data_to_metrics(input_csv_file, save_path, project_name)
    print(f"指标转换完成，结果保存至：{output_file}")