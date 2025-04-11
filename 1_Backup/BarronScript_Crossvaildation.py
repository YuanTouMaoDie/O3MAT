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
    df_obs = pd.read_csv(monitor_file)
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None

    # 根据日期或日期范围过滤监测数据
    if date:
        df_obs = df_obs[df_obs["Date"] == date]
    elif date_range:
        df_obs = df_obs[(df_obs["Date"] >= date_range[0]) & (df_obs["Date"] <= date_range[1])]

    # 根据交叉验证类型处理数据
    if cv_type == 'kfold':
        #random_state为随机种子
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(df_obs), 1):
            df_train = df_obs.iloc[train_index]
            df_test = df_obs.iloc[test_index]
            df_result = process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold)
            df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
    elif cv_type == 'custom':
        df_train = df_obs[df_obs["Site"].isin(train_sites)]
        df_test = df_obs[df_obs["Site"].isin(test_sites)]
        df_result = process_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold=0)  # 自定义分组交叉验证设为 0
        df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
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
                if isinstance(ds_model['TSTEP'].values[0], np.int64):
                    timeIndex = get_day_of_year(date) - 1
                    ds_daily_model = ds_model.sel(TSTEP=timeIndex)
                else:
                    ds_daily_model = ds_model.sel(TSTEP=date)

                df_daily_obs["x"], df_daily_obs["y"] = proj(df_daily_obs["Lon"], df_daily_obs["Lat"])
                df_daily_obs["mod"] = ds_daily_model[model_pollutant][0].sel(
                    ROW=df_daily_obs["y"].to_xarray(),
                    COL=df_daily_obs["x"].to_xarray(),
                    method="nearest"
                )
                df_daily_obs["bias"] = df_daily_obs["mod"] - df_daily_obs["Conc"]
                df_daily_obs["r_n"] = df_daily_obs["Conc"] / df_daily_obs["mod"]

                df_prediction = ds_daily_model[["ROW", "COL"]].to_dataframe().reset_index()
                nn.fit(
                    df_daily_obs[["x", "y"]],
                    df_daily_obs[[monitor_pollutant, "mod", "bias", "r_n"]]
                )

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
                df_fusion["Fold"] = 0  # 无交叉验证设为 0

                if df_all_daily_prediction is None:
                    df_all_daily_prediction = df_fusion
                else:
                    df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_fusion])

                end_time = time.time()
                duration = end_time - start_time
                print(f"Data Fusion for {date} took {duration:.2f} seconds")

    if df_all_daily_prediction is not None:
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")
    else:
        print("没有有效的融合数据，未保存结果。")

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

    df_train_grouped = (
        df_train.groupby(["Site", "Date"])
       .aggregate({"Conc": "mean", "Lat": "mean", "Lon": "mean"})
       .sort_values(by="Date")
    ).reset_index()
    dates = df_train_grouped["Date"].unique()

    for date in dates:
        df_daily_train = df_train_grouped[df_train_grouped["Date"] == date].copy()
        if isinstance(ds_model['TSTEP'].values[0], np.int64):
            timeIndex = get_day_of_year(date) - 1
            ds_daily_model = ds_model.sel(TSTEP=timeIndex)
        else:
            ds_daily_model = ds_model.sel(TSTEP=date)

        df_daily_train["x"], df_daily_train["y"] = proj(df_daily_train["Lon"], df_daily_train["Lat"])
        df_daily_train["mod"] = ds_daily_model[model_pollutant][0].sel(
            ROW=df_daily_train["y"].to_xarray(),
            COL=df_daily_train["x"].to_xarray(),
            method="nearest"
        )
        df_daily_train["bias"] = df_daily_train["mod"] - df_daily_train["Conc"]
        df_daily_train["r_n"] = df_daily_train["Conc"] / df_daily_train["mod"]

        nn.fit(
            df_daily_train[["x", "y"]],
            df_daily_train[[monitor_pollutant, "mod", "bias", "r_n"]]
        )

        df_daily_test = df_test[df_test["Date"] == date].copy()
        df_daily_test["x"], df_daily_test["y"] = proj(df_daily_test["Lon"], df_daily_test["Lat"])
        df_daily_test["mod"] = ds_daily_model[model_pollutant][0].sel(
            ROW=df_daily_test["y"].to_xarray(),
            COL=df_daily_test["x"].to_xarray(),
            method="nearest"
        )
        df_daily_test["bias"] = df_daily_test["mod"] - df_daily_test["Conc"]
        df_daily_test["r_n"] = df_daily_test["Conc"] / df_daily_test["mod"]

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
        df_fusion["Fold"] = fold  # 添加折数信息

        # 标记测试集的监测值所属的网格
        for index, row in df_daily_test.iterrows():
            grid_row = int((row["y"] + 0.5))
            grid_col = int((row["x"] + 0.5))
            df_fusion.loc[(df_fusion["ROW"] == grid_row) & (df_fusion["COL"] == grid_col), "Observation"] = row["Conc"]

        if df_result is None:
            df_result = df_fusion
        else:
            df_result = pd.concat([df_result, df_fusion])

    return df_result


def calculate_metrics(df, timestamp=None, fold=None):
    """
    计算监测值和相应的 vna、evna、avna 和 mod 的 RMSE、MB、R-squared、Slope。
    :param df: 输入的数据框
    :param timestamp: 指定的时间戳
    :param fold: 指定的折数
    :return: 包含指标结果的字典
    """
    if timestamp:
        df = df[df['Timestamp'] == timestamp]
    if fold:
        df = df[df['Fold'] == fold]

    df = df.dropna(subset=['Observation'])
    columns = ['vna_ozone', 'evna_ozone', 'avna_ozone', 'model']
    metrics = {}
    for col in columns:
        y_true = df['Observation']
        y_pred = df[col]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mb = np.mean(y_pred - y_true)
        r2 = r2_score(y_true, y_pred)
        slope, _, _, _, _ = linregress(y_true, y_pred)
        metrics[f'RMSE_{col.upper()}'] = rmse
        metrics[f'MB_{col.upper()}'] = mb
        metrics[f'R-squared_{col.upper()}'] = r2
        metrics[f'Slope_{col.upper()}'] = slope
    return metrics


def calculate_all_metrics(df, calculate_fold_mean=True, calculate_Timestamp_mean=True):
    """
    计算所有组合的指标，包括具体组合以及按 Fold 和 Timestamp 的平均值（如果需要）。
    :param df: 输入的数据框
    :param calculate_fold_mean: 是否计算按 Fold 的平均值，默认为 True
    :param calculate_Timestamp_mean: 是否计算按 Timestamp 的平均值，默认为 True
    :return: 包含所有指标结果的 DataFrame
    """
    all_timestamps = df['Timestamp'].unique()
    all_folds = df['Fold'].unique()
    results = []

    # 遍历每个 Timestamp 和 Fold 组合
    for timestamp in all_timestamps:
        for fold in all_folds:
            metrics = calculate_metrics(df, timestamp, fold)
            result = {
                'Timestamp': timestamp,
                'Fold': fold
            }
            result.update(metrics)
            results.append(result)

    if calculate_Timestamp_mean:
        # 计算每个 Timestamp 的平均值
        for timestamp in all_timestamps:
            metrics = calculate_metrics(df, timestamp)
            result = {
                'Timestamp': timestamp,
                'Fold': 'Mean'
            }
            result.update(metrics)
            results.append(result)

    if calculate_fold_mean:
        # 计算每个 Fold 的平均值
        for fold in all_folds:
            metrics = calculate_metrics(df, fold=fold)
            result = {
                'Timestamp': 'Mean',
                'Fold': fold
            }
            result.update(metrics)
            results.append(result)

        # 计算所有数据的平均值
        metrics = calculate_metrics(df)
        result = {
            'Timestamp': 'Mean',
            'Fold': 'Mean'
        }
        result.update(metrics)
        results.append(result)

    return pd.DataFrame(results)

#先平均然后再取出监测数据的训练集
def process_data_average(model_file, monitor_file, file_path, dict_period, monitor_pollutant="ozone",
                            model_pollutant="O3", cv_type=None, k=5, train_sites=None, test_sites=None):
    """
    @param {string} model_file: 模型文件，必须有维度：Time, Layer, ROW, COL，以及变量：O3_MDA8
    @param {string} monitor_file: 监测文件，必须有列：Site, POC, Date, Lat, Lon, Conc
    @param {string} file_path: 输出文件路径
    @param {dict} dict_period: 每个数据融合的时间段，键是时间段名称，值是时间段的开始和结束日期列表
    @param {string} cv_type: 交叉验证类型，可选值为 'kfold' 或 'custom'，默认为 None
    @param {int} k: k折交叉验证的折数，默认为 5
    @param {list} train_sites: 自定义分组交叉验证时的训练集站点列表，默认为 None
    @param {list} test_sites: 自定义分组交叉验证时的测试集站点列表，默认为 None
    """
    ds_model = pyrsig.open_ioapi(model_file)
    proj = pyproj.Proj(ds_model.crs_proj4)
    df_obs = pd.read_csv(monitor_file)
    # 控制变量
    df_obs = df_obs[df_obs['Site'] != 60650008]
    nn = nna_methods.NNA(method="voronoi", k=30)
    df_all_daily_prediction = None
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])

    with tqdm(dict_period.items()) as pbar:
        for period_name, period in pbar:
            start_date, end_date = period[0], period[1]
            pbar.set_description(f"Data Fusion for {period_name}...")

            if start_date > end_date:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) | (df_obs["Date"] <= end_date)]
            else:
                df_filtered_obs = df_obs[(df_obs["Date"] >= start_date) & (df_obs["Date"] <= end_date)]

            df_avg_obs = (
                df_filtered_obs.groupby(["Site"]).aggregate(
                    {"Conc": "mean", "Lat": "mean", "Lon": "mean"}
                )
            ).reset_index()

            # 根据交叉验证类型处理数据
            if cv_type == 'kfold':
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                for fold, (train_index, test_index) in enumerate(kf.split(df_avg_obs), 1):
                    df_train = df_avg_obs.iloc[train_index]
                    df_test = df_avg_obs.iloc[test_index]
                    df_result = process_fold_data(df_train, df_test, ds_model, proj, nn, model_pollutant,
                                                monitor_pollutant, fold, start_date, end_date, period_name)
                    df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
            elif cv_type == 'custom':
                df_train = df_avg_obs[df_avg_obs["Site"].isin(train_sites)]
                df_test = df_avg_obs[df_avg_obs["Site"].isin(test_sites)]
                df_result = process_fold_data(df_train, df_test, ds_model, proj, nn, model_pollutant,
                                            monitor_pollutant, 0, start_date, end_date, period_name)
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result
            else:
                df_result = process_fold_data(df_avg_obs, None, ds_model, proj, nn, model_pollutant,
                                            monitor_pollutant, 0, start_date, end_date, period_name)
                df_all_daily_prediction = pd.concat([df_all_daily_prediction, df_result]) if df_all_daily_prediction is not None else df_result

    if df_all_daily_prediction is not None:
        df_all_daily_prediction.to_csv(file_path, index=False)
        print(f"Data Fusion for all dates is done, the results are saved to {file_path}")


    #先平均后融合的处理函数
def process_fold_data(df_train, df_test, ds_model, proj, nn, model_pollutant, monitor_pollutant, fold, start_date,
                        end_date, period_name):
    if isinstance(ds_model["TSTEP"].values[0], np.int64):
        start_time_index = get_day_of_year(start_date) - 1
        end_time_index = get_day_of_year(end_date) - 1
        if start_date > end_date:
            ds_avg_model = ds_model.sel(
                TSTEP=list(range(start_time_index, len(ds_model["TSTEP"]))) + list(range(0, end_time_index + 1))
            ).mean(dim="TSTEP")
        else:
            ds_avg_model = ds_model.sel(
                TSTEP=slice(start_time_index, end_time_index)
            ).mean(dim="TSTEP")
    else:
        if start_date > end_date:
            ds_avg_model = ds_model.sel(
                TSTEP=list(pd.date_range(start=start_date, end='2011-12-31')) + list(
                    pd.date_range(start='2011-01-01', end=end_date))
            ).mean(dim="TSTEP")
        else:
            ds_avg_model = ds_model.sel(TSTEP=slice(start_date, end_date)).mean(dim="TSTEP")

    df_train["x"], df_train["y"] = proj(df_train["Lon"], df_train["Lat"])
    df_train["mod"] = ds_avg_model[model_pollutant][0].sel(
        ROW=df_train["y"].to_xarray(),
        COL=df_train["x"].to_xarray(),
        method="nearest"
    )
    df_train["bias"] = df_train["mod"] - df_train["Conc"]
    df_train["r_n"] = df_train["Conc"] / df_train["mod"]

    nn.fit(
        df_train[["x", "y"]],
        df_train[[monitor_pollutant, "mod", "bias", "r_n"]]
    )

    df_prediction = ds_avg_model[["ROW", "COL"]].to_dataframe().reset_index()
    zdf = nn.predict(df_prediction[["COL", "ROW"]].values)
    df_prediction["vna_ozone"], df_prediction["vna_mod"], df_prediction["vna_bias"], df_prediction["vna_r_n"] = zdf.T

    df_fusion = df_prediction.set_index(["ROW", "COL"]).to_xarray()
    df_fusion["avna_ozone"] = ds_avg_model[model_pollutant][0].values - df_fusion["vna_bias"]
    reshaped_vna_r_n = df_prediction["vna_r_n"].values.reshape(ds_avg_model[model_pollutant][0].shape)
    df_fusion["evna_ozone"] = (("ROW", "COL"), ds_avg_model[model_pollutant][0].values * reshaped_vna_r_n)
    df_fusion = df_fusion.to_dataframe().reset_index()
    df_fusion["model"] = ds_avg_model[model_pollutant][0].values.flatten()
    df_fusion["Period"] = period_name
    df_fusion["COL"] = (df_fusion["COL"] + 0.5).astype(int)
    df_fusion["ROW"] = (df_fusion["ROW"] + 0.5).astype(int)
    df_fusion["Fold"] = fold

    if df_test is not None:
        df_test["x"], df_test["y"] = proj(df_test["Lon"], df_test["Lat"])
        df_test["mod"] = ds_avg_model[model_pollutant][0].sel(
            ROW=df_test["y"].to_xarray(),
            COL=df_test["x"].to_xarray(),
            method="nearest"
        )
        for index, row in df_test.iterrows():
            grid_row = int((row["y"] + 0.5))
            grid_col = int((row["x"] + 0.5))
            df_fusion.loc[(df_fusion["ROW"] == grid_row) & (df_fusion["COL"] == grid_col), "Observation"] = row["Conc"]

    return df_fusion



if __name__ == "__main__":
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file_name = "BarronHarvard_20110101_ALL_CV.csv"
    output_file_path = os.path.join(save_path, output_file_name)
    seasonal_output_path = os.path.join(save_path, "BarronHarvard_2011_ALL_AtFAnnual_CV.csv")

    model_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/HR2DAY_LST_ACONC_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2011_12US1_2011.nc"
    # model_file = r"/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc"
    monitor_file = r"/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv"

    # # 示例：使用k折交叉验证，指定日期范围，随机分折
    # output_file = start_daily_data_fusion(
    #     model_file,
    #     monitor_file,
    #     output_file_path,
    #     monitor_pollutant="Conc",
    #     model_pollutant="MDA8_O3",
    #     cv_type='kfold',
    #     k=10,
    #     date_range=['2011-01-01', '2011-01-01'],
    #     shuffle=True
    # )
        # 示例：使用k折交叉验证
    process_data_average(
        model_file,
        monitor_file,
        seasonal_output_path,
        monitor_pollutant="Conc",
        model_pollutant="O3_MDA8",
        cv_type='kfold',
        k=10,
        dict_period={
            # "DJF_2011": ["2011-12-01", "2011-02-28"],
            # "MAM_2011": ["2011-03-01", "2011-05-31"],
            # "JJA_2011": ["2011-06-01", "2011-08-31"],
            # "SON_2011": ["2011-09-01", "2011-11-30"],
            "Annual_2011": ["2011-01-01", "2011-12-31"],
            # "Apr-Sep_2011": ["2011-04-01", "2011-09-30"],
        }
    )
    print("Done!")

    # # 读取输出文件
    # df = pd.read_csv(output_file)

    # # 调用集成函数计算所有指标，可根据需要设置是否计算 Fold 和 Timestamp 的平均值
    # result_df = calculate_all_metrics(df, calculate_fold_mean=True, calculate_Timestamp_mean=False)

    # 保存结果到文件
    # result_file_path = os.path.join(save_path, "Metrics_Results.csv")
    # result_df.to_csv(result_file_path, index=False)
    # print(f"指标计算结果已保存到 {result_file_path}")
