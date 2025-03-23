import pyrsig
import pyproj
import nna_methods
import os
from tqdm.auto import tqdm
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

cmap_conc = cmaps.WhiteBlueGreenYellowRed
cmap_delta = cmaps.ViBlGrWhYeOrRe


def process_daily_data(df_daily_obs, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold):
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
    zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
    df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

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
def start_daily_data_fusion(model_file, monitor_file, file_path, monitor_pollutant="ozone", model_pollutant="O3",
                            cv_type=None, k=5, train_sites=None, test_sites=None, date=None, date_range=None, shuffle=True):
    """
    进行每日数据融合操作。
    :param model_file: 模型文件路径，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    :param monitor_file: 监测文件路径，必须有列：Site, POC, Date, Lat, Lon, Conc
    :param file_path: 输出文件路径
    :param monitor_pollutant: 监测文件中的污染物，默认是 ozone
    :param model_pollutant: 模型文件中的污染物，默认是 O3
    :param cv_type: 交叉验证类型，可选值为 'kfold' 或 'custom'，默认为 None
    :param k: k折交叉验证的折数，默认为 5
    :param train_sites: 自定义分组交叉验证时的训练集站点列表，默认为 None
    :param test_sites: 自定义分组交叉验证时的测试集站点列表，默认为 None
    :param date: 单个日期，格式为 'YYYY-MM-DD'，默认为 None
    :param date_range: 日期范围，格式为 ['YYYY-MM-DD', 'YYYY-MM-DD']，默认为 None
    :param shuffle: 分折时是否随机打乱数据，默认为 True
    :return: 输出文件路径
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file).dropna(subset=['Conc', 'Lat', 'Lon'])  # 处理缺失值
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    all_metrics = []

    # 根据日期或日期范围过滤监测数据
    if date:
        df_obs = df_obs[df_obs["Date"] == date]
    elif date_range:
        df_obs = df_obs[(df_obs["Date"] >= date_range[0]) & (df_obs["Date"] <= date_range[1])]

    # 根据交叉验证类型处理数据
    if cv_type == 'kfold':
        # random_state为随机种子
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(df_obs), 1):
            df_train = df_obs.iloc[train_index].copy()
            df_test = df_obs.iloc[test_index].copy()
            df_result, metrics = process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold)
            df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
            # 遍历 metrics 列表，为每个字典添加 Fold 信息
            for metric in metrics:
                metric['Fold'] = fold
            all_metrics.extend(metrics)
    elif cv_type == 'custom':
        df_train = df_obs[df_obs["Site"].isin(train_sites)].copy()
        df_test = df_obs[df_obs["Site"].isin(test_sites)].copy()
        df_result, metrics = process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold=0)  # 自定义分组交叉验证设为 0
        df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
        # 遍历 metrics 列表，为每个字典添加 Fold 信息
        for metric in metrics:
            metric['Fold'] = 0
        all_metrics.extend(metrics)
    else:
        df_obs_grouped = (
            df_obs.groupby(["Site", "Date"])
           .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
           .sort_values(by="Date")
        ).reset_index()
        dates = df_obs_grouped["Date"].unique()

        with tqdm(dates) as pbar:
            for date in pbar:
                pbar.set_description(f"Data Fusion for {date}...")
                start_time = time.time()

                df_daily_obs = df_obs_grouped[df_obs_grouped["Date"] == date].copy()
                df_fusion, _ = process_daily_data(df_daily_obs, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, 0)

                if df_all_daily_prediction is None:
                    df_all_daily_prediction = df_fusion
                else:
                    df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

                end_time = time.time()
                duration = end_time - start_time
                print(f"Data Fusion for {date} took {duration:.2f} seconds")

                # 计算评估指标
                metrics = calculate_metrics_single_day(df_fusion, df_daily_obs, monitor_pollutant, date, 0)
                all_metrics.append(metrics)

    if df_all_daily_prediction is not None:
        df_all_daily_prediction = df_all_daily_prediction.drop(columns=['Observation'], errors='ignore')
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")

    metrics_df = pd.DataFrame(all_metrics)
    # 计算所有折的平均值
    mean_metrics = metrics_df.drop(columns=['Timestamp', 'Fold']).mean()
    mean_metrics['Timestamp'] = 'Mean'
    mean_metrics['Fold'] = 'All'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_metrics])], ignore_index=True)

    metrics_file_path = file_path.replace('.csv', '_metrics.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"Metrics calculation results are saved to {metrics_file_path}")

    return file_path


def process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold):
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
    :return: 处理后的结果数据框
    """
    df_result = None
    all_metrics = []

    df_train_grouped = (
        df_train.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    dates = df_train_grouped["Date"].unique()

    for date in dates:
        df_daily_train = df_train_grouped[df_train_grouped["Date"] == date].copy()
        df_fusion, _ = process_daily_data(df_daily_train, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold)

        df_daily_test = df_test[df_test["Date"] == date].copy()
        # 对 df_daily_test 进行坐标转换，添加 x 和 y 列
        df_daily_test["x"], df_daily_test["y"] = proj(df_daily_test["Lon"], df_daily_test["Lat"])
        # 处理测试集数据，记录匹配信息
        _, df_daily_test = process_daily_data(df_daily_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, date, fold)

        # 计算评估指标
        metrics = calculate_metrics_single_day(df_fusion, df_daily_test, monitor_pollutant, date, fold)
        all_metrics.append(metrics)

        if df_result is None:
            df_result = df_fusion
        else:
            df_result = pd.concat([df_result, df_fusion])

    return df_result, all_metrics


def calculate_metrics_single_day(df_fusion, df_test, monitor_pollutant, date, fold):
    columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    metrics = {
        'Timestamp': date,
        'Fold': fold
    }
    unmatched_rows = []
    for col in columns:
        y_true = df_test[monitor_pollutant]
        y_pred = []
        for i, (row, col_val) in enumerate(zip(df_test["matched_ROW"], df_test["matched_COL"])):
            # 对 ROW 和 COL 加 0.5 并取整
            row_rounded = int(row + 0.5)
            col_val_rounded = int(col_val + 0.5)
            match = df_fusion[(df_fusion["ROW"] == row_rounded) & (df_fusion["COL"] == col_val_rounded)][col]
            if not match.empty:
                y_pred.append(match.values[0])
            else:
                print(f"Warning: No match found for ROW {row_rounded}, COL {col_val_rounded} (index {i}) on date {date}, Fold {fold}")
                unmatched_rows.append(i)
        
        #自动根据索引对齐函数
        y_pred = pd.Series(y_pred)

        # 处理不匹配的情况，比如Harvard数据集中的部分站点没有匹配到模型网格点
        if len(y_true) != len(y_pred):
            print(f"Length mismatch between y_true and y_pred for {col} on date {date}, Fold {fold}")
            print(f"Indices of unmatched rows: {unmatched_rows}")
            print(f"Unmatched rows in df_test:\n{df_test.iloc[unmatched_rows]}")
        # 模型网格域全局有数据，对于EQUATES一定成立
        if len(y_true) == len(y_pred):
            try:
                # 重置索引
                y_true = y_true.reset_index(drop=True)
                y_pred = y_pred.reset_index(drop=True)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                print(f"For {col} on date {date}, Fold {fold}:")
                diff = y_pred - y_true
                mb = np.mean(diff)
                r2 = r2_score(y_true, y_pred)
                slope, _, _, _, _ = linregress(y_true, y_pred)
                metrics[f'RMSE_{col.upper()}'] = rmse
                metrics[f'MB_{col.upper()}'] = mb
                metrics[f'R-squared_{col.upper()}'] = r2
                metrics[f'Slope_{col.upper()}'] = slope
            except Exception as e:
                print(f"Error calculating metrics for {col} on date {date}, Fold {fold}: {e}")
        else:
            print(f"Cannot calculate metrics for {col} on date {date}, Fold {fold} due to length mismatch.")

    return metrics


if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "BarronScript_2011_ALL_FtA0101.csv"
    output_file_path = os.path.join(save_path, output_file_name)

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    # model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # 示例：使用k折交叉验证，指定日期范围，随机分折
    output_file = start_daily_data_fusion(
        model_file,
        monitor_file,
        output_file_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        cv_type='kfold',
        k=10,
        date_range=['2011-01-01', '2011-01-01'],
        shuffle=True
    )