import pyrsig
import pyproj
import nna_methods
import os
import pandas as pd
import time
import numpy as np
from esil.date_helper import timer_decorator, get_day_of_year
from esil.rsm_helper.model_property import model_attribute
from esil.map_helper import get_multiple_data, show_maps
import cmaps
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from sklearn.model_selection import KFold
import multiprocessing
import matplotlib.pyplot as plt

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


def process_daily_data(df_daily_obs, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold, region_table_file):
    """
    处理每日数据的函数，将重复的处理逻辑封装在此
    """
    if isinstance(ds_model['TSTEP'].values[0], np.int64):
        timeIndex = get_day_of_year(date) - 1
        ds_daily_model = ds_model.sel(TSTEP=timeIndex)
    else:
        ds_daily_model = ds_model.sel(TSTEP=date)

    df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])
    # 记录匹配的模型网格点信息
    matched = ds_daily_model[model_pollutant][0].sel(
        ROW=df_daily_obs["y"].to_xarray(),
        COL=df_daily_obs["x"].to_xarray(),
        method="nearest",
        drop=True
    )
    df_daily_obs["mod"] = matched
    df_daily_obs["matched_ROW"] = matched.ROW.values
    df_daily_obs["matched_COL"] = matched.COL.values

    # 检查匹配信息的长度
    if len(df_daily_obs) != len(df_daily_obs["matched_ROW"]):
        print(f"Warning: Length mismatch between df_daily_obs and matched_ROW on date {date}, Fold {fold}")
    if len(df_daily_obs) != len(df_daily_obs["matched_COL"]):
        print(f"Warning: Length mismatch between df_daily_obs and matched_COL on date {date}, Fold {fold}")

    df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
    df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

    nn.fit(
        df_daily_obs[["x", "y"]],
        df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
    )

    df_prediction = ds_daily_model[["ROW", "COL"]].to_dataframe().reset_index()

    # 读取包含 Is 列的数据表
    try:
        region_df = pd.read_csv(region_table_file)
        # 筛选出 Is 列值为 1 的行
        us_region_df = region_df[region_df['Is'] == 1]
        # -0.5确保后面的计算是正确的
        us_region_df[['COL', 'ROW']] = us_region_df[['COL', 'ROW']] - 0.5
        # 输入为x,y形式，对于model的col和row
        us_region_row_col = us_region_df[['COL', 'ROW']].values

        # 并行计算部分
        njobs = multiprocessing.cpu_count()  # 使用所有CPU核心进行并行计算
        zdf = nn.predict(us_region_row_col, njobs=njobs)

        # 创建一个全为 NaN 的 DataFrame 用于存储预测结果
        result_df = pd.DataFrame(np.nan, index=df_prediction.index, columns=["vna_ozone", "vna_mod", "vna_bias", "vna_r_n"])
        # 将美国区域的预测结果填充到对应的行
        result_df.loc[us_region_df.index] = zdf

        df_prediction = pd.concat([df_prediction, result_df], axis=1)
    except FileNotFoundError:
        print(f"Error: Region table file {region_table_file} not found. Skipping region-based fusion.")
        df_prediction["vna_ozone"] = np.nan
        df_prediction["vna_mod"] = np.nan
        df_prediction["vna_bias"] = np.nan
        df_prediction["vna_r_n"] = np.nan

    df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
    df_fusion["avna_ozone"] = ds_daily_model[model_pollutant][0].values - df_fusion["vna_bias"]
    reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_daily_model[model_pollutant][0].shape)
    df_fusion["evna_ozone"] = (("ROW", "COL"), ds_daily_model[model_pollutant][0].values * reshaped_vna_r_n)
    df_fusion = df_fusion.to_dataframe().reset_index()
    df_fusion["model"] = ds_daily_model[model_pollutant][0].values.flatten()
    df_fusion["Timestamp"] = date
    df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
    df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
    df_fusion["Fold"] = fold

    return df_fusion, df_daily_obs


@timer_decorator
def start_daily_data_fusion(model_file, monitor_file, file_path, region_table_file,
                            monitor_pollutant="ozone", model_pollutant="O3",
                            k=5, date=None, date_range=None, shuffle=True,
                            calculate_metrics=True):
    """
    进行每日数据融合操作。
    :param model_file: 模型文件路径，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    :param monitor_file: 监测文件路径，必须有列：Site, POC, Date, Lat, Lon, Conc
    :param file_path: 输出文件路径
    :param region_table_file: 包含 Is 列的数据表文件路径
    :param monitor_pollutant: 监测文件中的污染物，默认是 ozone
    :param model_pollutant: 模型文件中的污染物，默认是 O3
    :param k: k折交叉验证的折数，默认为 5
    :param date: 单个日期，格式为 'YYYY-MM-DD'，默认为 None
    :param date_range: 日期范围，格式为 ['YYYY-MM-DD', 'YYYY-MM-DD']，默认为 None
    :param shuffle: 分折时是否随机打乱数据，默认为 True
    :param calculate_metrics: 是否计算评估指标，默认为 True
    :return: 输出文件路径，交叉验证结果数据框
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file).dropna(subset=['Conc', 'Lat', 'Lon'])  # 处理缺失值
    nn = nna_methods.NNA(method="voronoi", k=30)
    all_results = []

    # 读取交叉验证设置
    cv_file = "/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"
    try:
        cv_df = pd.read_csv(cv_file)
        if date:
            cv_df = cv_df[cv_df['Date'] == date]
        elif date_range:
            cv_df = cv_df[cv_df['Date'].between(date_range[0], date_range[1])]

        # 从cv_df中获取每个站点的CVgroup
        cv_groups = cv_df.set_index('Site')['CVgroup'].to_dict()
        df_obs = df_obs[df_obs['Date'].isin(cv_df['Date'])]  # 确保 df_obs 只包含指定日期的数据
        df_obs['CVgroup'] = df_obs['Site'].map(cv_groups)
    except FileNotFoundError:
        print(f"Error: CV settings file {cv_file} not found. Using default cross-validation.")
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)

    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    if date_range:
        dates = pd.date_range(date_range[0], date_range[1]).strftime('%Y-%m-%d')
    elif date:
        dates = [date]
    else:
        dates = df_obs_grouped["Date"].unique()

    # 用于存储所有折的每日预测结果
    all_daily_prediction_dfs = []

    # K折交叉验证
    try:
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
        for date in dates:
            print(f"Processing date {date}")
            df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
            if df_daily_obs.empty:
                print(f"No data available for date {date}. Skipping...")
                continue
            daily_prediction_dfs = []
            for fold, (train_index, test_index) in enumerate(kf.split(df_obs), 1):
                fold_start_time = time.time()
                print(f"Processing fold {fold} for {date}")
                df_train = df_obs.iloc[train_index].copy()
                df_test = df_obs.iloc[test_index].copy()
                df_result, results = process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold, date, region_table_file)
                daily_prediction_dfs.append(df_result)
                all_results.extend(results)
                fold_end_time = time.time()
                fold_duration = fold_end_time - fold_start_time
                print(f"Fold {fold} for {date} took {fold_duration:.2f} seconds")
            daily_prediction = pd.concat(daily_prediction_dfs)
            all_daily_prediction_dfs.append(daily_prediction)
            # 每天的数据处理完后保存到文件
            daily_file_path = file_path.replace('.csv', f'_{date}.csv')
            daily_prediction.to_csv(daily_file_path, index=False)
            print(f"Data Fusion for {date} is done, the results are saved to {daily_file_path}")
    except Exception as e:
        print(f"Error during cross-validation: {e}")

    # 合并所有日期的结果
    if all_daily_prediction_dfs:
        df_all_daily_prediction = pd.concat(all_daily_prediction_dfs)
        df_all_daily_prediction = df_all_daily_prediction.drop(columns=['Observation'], errors='ignore')
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

    results_df = pd.DataFrame(all_results)
    # 修正年份提取逻辑
    year = os.path.basename(monitor_file).split('.')[-2]
    cv_data_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/{year}_Data_CV"
    if not os.path.exists(cv_data_path):
        os.makedirs(cv_data_path)
    results_file_name = os.path.basename(file_path).replace('.csv', '_CV_results.csv')
    results_file_path = os.path.join(cv_data_path, results_file_name)
    results_df.to_csv(results_file_path, index=False)
    print(f"Cross-validation results are saved to {results_file_path}")

    return file_path, results_df


def process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold, date, region_table_file):
    """
    处理训练集和测试集数据的辅助函数。
    :param df_train: 训练集数据框
    :param df_test: 测试集数据框
    :param ds_model: 模型数据集
    :param proj: 投影对象
    :param nn: 近邻分析对象
    :param model_pollutant: 模型文件中的污染物
    :param monitor_pollutant: 监测文件中的污染物
    :param fold: 当前折数
    :param date: 当前日期
    :param region_table_file: 包含 Is 列的数据表文件路径
    :return: 处理后的结果数据框
    """
    df_result = None
    all_results = []

    df_train_grouped = (
        df_train.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    df_daily_train = df_train_grouped[df_train_grouped["Date"] == date].copy()
    df_fusion, _ = process_daily_data(df_daily_train, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold, region_table_file)

    df_daily_test = df_test[df_test["Date"] == date].copy()
    # 对 df_daily_test 进行坐标转换，添加 x 和 y 列
    df_daily_test["x"], df_daily_test["y"] = proj(df_daily_test["Lon"], df_daily_test["Lat"])
    # 处理测试集数据，记录匹配信息
    _, df_daily_test = process_daily_data(df_daily_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold, region_table_file)

    results = get_results(df_fusion, df_daily_test, monitor_pollutant, date, fold)
    all_results.extend(results)

    return df_fusion, all_results


def get_results(df_fusion, df_test, monitor_pollutant, date, fold):
    columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    results = []
    for col in columns:
        y_true = df_test[monitor_pollutant]
        for i, (row, col_val) in enumerate(zip(df_test["matched_ROW"], df_test["matched_COL"])):
            # 对 ROW 和 COL 加 0.5 并取整
            row_rounded = int(row + 0.5)
            col_val_rounded = int(col_val + 0.5)
            match = df_fusion[(df_fusion["ROW"] == row_rounded) & (df_fusion["COL"] == col_val_rounded)][col]
            if not match.empty:
                result = {
                    'Timestamp': date,
                    'Fold': fold,
                    'Method': col,
                    'y_true': y_true.iloc[i],
                    'y_pred': match.values[0],
                    'ROW': row_rounded,
                    'COL': col_val_rounded
                }
                results.append(result)
    return results


def plot_scatter(results_df, monitor_pollutant, file_path):
    methods = results_df['Method'].unique()
    # 修正年份提取逻辑
    year = os.path.basename(file_path).split('.')[0].split('_')[-2]
    cv_picture_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/{year}_Picture_CV"
    if not os.path.exists(cv_picture_path):
        os.makedirs(cv_picture_path)

    # 获取 Timestamp 信息
    timestamps = results_df['Timestamp'].unique()
    if timestamps.size > 0:
        timestamp = timestamps[0]
    else:
        timestamp = "NoTimestamp"

    for method in methods:
        subset = results_df[results_df['Method'] == method]
        y_true = subset['y_true']
        y_pred = subset['y_pred']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        diff = y_pred - y_true
        mb = np.mean(diff)
        r2 = r2_score(y_true, y_pred)
        slope, intercept, _, _, _ = linregress(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)

        # 添加 y = x 的蓝线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='blue')

        plt.plot(y_true, slope * y_true + intercept, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

        # 设置图标题
        if method.lower().startswith('vna'):
            method_display = 'VNA'
        elif method.lower().startswith('evna'):
            method_display = 'eVNA'
        elif method.lower().startswith('avna'):
            method_display = 'aVNA'
        else:
            method_display = method.upper()
        plt.title(f'{timestamp}_{method_display} vs. Monitor')

        # 设置 x 轴标签为 Monitor
        plt.xlabel('Monitor')

        # 根据 Method 设置 y 轴标签
        plt.ylabel(method_display)

        plt.legend()

        # 调整 RMSE、MB 和 R-squared 文字的字号，调大 2 号
        current_font_size = plt.rcParams.get('font.size', 10)
        new_font_size = current_font_size + 2
        plt.text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=new_font_size)
        plt.text(0.05, 0.85, f'MB: {mb:.2f}', transform=plt.gca().transAxes, fontsize=new_font_size)
        # 使用 LaTeX 语法显示 R²
        plt.text(0.05, 0.8, f'$R^{2}$: {r2:.2f}', transform=plt.gca().transAxes, fontsize=new_font_size)

        # 让横纵轴范围一致
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # 修改输出文件名
        plot_file_name = f'{timestamp}_{method_display}_{monitor_pollutant}_scatter_plot.png'
        plot_file_path = os.path.join(cv_picture_path, plot_file_name)
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Scatter plot for {timestamp}_{method_display} saved to {plot_file_path}")


def plot_scatter_from_file(results_file_path, monitor_pollutant):
    """
    从文件中读取交叉验证结果数据，并绘制散点图
    :param results_file_path: 交叉验证结果文件路径
    :param monitor_pollutant: 监测文件中的污染物
    """
    results_df = pd.read_csv(results_file_path)
    plot_scatter(results_df, monitor_pollutant, results_file_path)


def generate_result_table(model_file, monitor_file, region_table_file, monitor_pollutant="Conc", model_pollutant="O3_MDA8", k=10, date_range=['2011-01-01', '2011-01-01']):
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file).dropna(subset=['Conc', 'Lat', 'Lon'])

    # 读取交叉验证设置
    cv_file = "/backupdata/data_EPA/EQUATES/CVruns/ozone_2011_cdc_12km.csv"
    try:
        cv_df = pd.read_csv(cv_file)
        if date_range:
            cv_df = cv_df[cv_df['Date'].between(date_range[0], date_range[1])]
        cv_groups = cv_df.set_index('Site')['CVgroup'].to_dict()
        df_obs = df_obs[df_obs['Date'].isin(cv_df['Date'])]  # 确保 df_obs 只包含指定日期的数据
        df_obs['CVgroup'] = df_obs['Site'].map(cv_groups)
    except FileNotFoundError:
        print(f"Error: CV settings file {cv_file} not found.")

    df_obs_grouped = (
        df_obs.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    dates = pd.date_range(date_range[0], date_range[1]).strftime('%Y-%m-%d')

    all_results = []
    nn = nna_methods.NNA(method="voronoi", k=30)

    for date in dates:
        df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
        if df_daily_obs.empty:
            print(f"No data available for date {date}. Skipping...")
            continue
        df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])

        if isinstance(ds_model['TSTEP'].values[0], np.int64):
            timeIndex = get_day_of_year(date) - 1
            ds_daily_model = ds_model.sel(TSTEP=timeIndex)
        else:
            ds_daily_model = ds_model.sel(TSTEP=date)

        matched = ds_daily_model[model_pollutant][0].sel(
            ROW=df_daily_obs["y"].to_xarray(),
            COL=df_daily_obs["x"].to_xarray(),
            method="nearest",
            drop=True
        )
        df_daily_obs["Prediction"] = matched

        for _, row in df_daily_obs.iterrows():
            result = {
                'Date': row['Date'],
                'Site': row['Site'],
                'POC': df_obs[(df_obs['Site'] == row['Site']) & (df_obs['Date'] == row['Date'])]['POC'].values[0],
                'Lat': row['Lat'],
                'Lon': row['Lon'],
                'Conc': row['Conc'],
                'CVgroup': row['CVgroup'],
                'Prediction': row['Prediction']
            }
            all_results.append(result)

    result_df = pd.DataFrame(all_results)
    year = os.path.basename(monitor_file).split('.')[-2]
    output_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/{year}_ResultTable.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Result table saved to {output_path}")
    return output_path


if __name__ == "__main__":
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"
    year = os.path.basename(monitor_file).split('.')[-2]
    save_path = f"/DeepLearning/mnt/shixiansheng/data_fusion/output/{year}_Data_CV"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "20110101_FourDataset_CV.csv"
    output_file_path = os.path.join(save_path, output_file_name)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    # model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"

    region_table_file = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/Region_CONUSHarvard.csv"  # 请替换为实际的区域数据表文件路径

    # 示例：使用k折交叉验证，指定日期范围，随机分折
    output_file, results_df = start_daily_data_fusion(
        model_file,
        monitor_file,
        output_file_path,
        region_table_file,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        k=10,
        date_range=['2011-01-01', '2011-01-01'],
        shuffle=True,
        calculate_metrics=True
    )

    # 选择是否绘制散点图
    plot_scatter_flag = True
    if plot_scatter_flag:
        plot_scatter(results_df, "Conc", output_file)

    # 生成结果表
    generate_result_table(model_file, monitor_file, region_table_file)
    