import os
import pyrsig
import pyproj
import nna_methods
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# 定义投影信息
proj_string = (
    "+proj=lcc "
    "+lat_0=40 +lon_0=-97 "
    "+lat_1=33 +lat_2=45 "
    "+x_0=2556000 +y_0=1728000 "
    "+R=6370000 "
    "+to_meter=12000 "
    "+no_defs"
)
proj = pyproj.Proj(proj_string)

# 添加输出文件名后缀生成函数
def generate_suffix(selected_months=None, selected_dates=None, year=None):
    if selected_months:
        if set(selected_months) == {1, 2, 3}:
            return f"JFM_{year}"
        elif set(selected_months) == {4, 5, 6}:
            return f"AMJ_{year}"
        elif set(selected_months) == {7, 8, 9}:
            return f"JAS_{year}"
        elif set(selected_months) == {10, 11, 12}:
            return f"OND_{year}"
        elif set(selected_months) == {4, 5, 6, 7, 8, 9}:
            return f"AS_{year}"
        elif set(selected_months) == {1,2,3,4,5,6,7,8,9,10,11,12}:
            return f"ALL_{year}"
        else:
            return f"{year}{selected_months}"
    elif selected_dates:
        return f"{selected_dates[0].strftime('%Y%m%d')}_{selected_dates[-1].strftime('%Y%m%d')}"
    return "UNSPECIFIED"

# 动态加载输入数据
def load_data(year, selected_months):
    
    # 监测数据路径
    monitor_path = f'D:/Project/DataFusion/DataFusion_MonitorDataInput_{year}/ds.input.aqs.o3.{year}.csv'
    if not os.path.exists(monitor_path):
        print(f"监测数据文件 {monitor_path} 不存在，跳过读取。")
        return None, None
    
    odf = pd.read_csv(monitor_path)
    odf['date'] = pd.to_datetime(odf['Date'])
    odf['month'] = odf['date'].dt.month

    #先筛选指定月份，再剔除掉无效的Conc值，再对同一个Site中同一天有多个数据的情况进行平均
    odf = odf[odf['month'].isin(selected_months)]
    odf = odf[(odf['Conc'] >= 0) & (~odf['Conc'].isna())]
    odf = odf.groupby(['Site', 'date'], as_index=False).agg({
        'Conc': 'mean',
        'Lat': 'first',
        'Lon': 'first',
        'month': 'first'
    })

    # 模型数据路径,同时读取多个月的模型数据
    eds_list = []
    for month in selected_months:
        model_path = f'D:/Project/DataFusion/DataFusion_ModelDataInput_{year}/Daily_EQUATES_CMAQv532_cb6r3_ae7_aq_STAGE_12US1_{year}{month:02d}.csv'
        if os.path.exists(model_path):
            #手动指定列名
            columns = [
                "column", "row", "longitude", "latitude", "Lambert_X", "LAMBERT_Y", "date",
                "O3_MDA8", "O3_AVG", "CO_AVG", "NO_AVG", "NO2_AVG", "SO2_AVG",
                "CH2O_AVG", "PM10_AVG", "PM25_AVG", "PM25_SO4", "PM25_NO3",
                "PM25_NH4", "PM25_OC", "PM25_EC"
            ]
            eds = pd.read_csv(model_path, skiprows=4, names=columns)
            eds['date'] = pd.to_datetime(eds['date'])
            eds_list.append(eds)
        else:
            print(f"模型数据文件 {model_path} 不存在，跳过读取。")
    
    if eds_list:
        eds_combined = pd.concat(eds_list, ignore_index=True)
        eds_combined['month'] = eds_combined['date'].dt.month
    else:
        print("未加载任何模型数据，退出。")
        return None, None

    return odf, eds_combined

# 处理函数，先均值后插值
def process_data_average(years, months_list):
 for year in years:
  for selected_months in months_list:
    print(f"正在处理年份: {year}, 月份: {selected_months}")
    odf, eds = load_data(year, selected_months)
    if odf is None or eds is None:
        print("数据加载失败，停止处理。")
        return

    suffix = generate_suffix(selected_months=selected_months, year=year)

    # 分组计算
    Model_O3_MDA8_avg = eds.groupby(['column', 'row']).agg({
        'O3_MDA8': 'mean',
        'longitude': 'first',
        'latitude': 'first',
        'Lambert_X': 'first',
        'LAMBERT_Y': 'first'
    }).reset_index()

    Monitor_O3_MDA8_avg = odf.groupby(['Site']).agg({
        'Conc': 'mean',
        'Lat': 'first',
        'Lon': 'first'
    }).reset_index()

    # 重命名并转换为 xarray
    odf = Monitor_O3_MDA8_avg
    eds = Model_O3_MDA8_avg
    eds = xr.Dataset.from_dataframe(Model_O3_MDA8_avg.set_index(['row', 'column']))
    eds = eds.rename({'row': 'ROW', 'column': 'COL'})
    odf['LONGITUDE'] = odf['Lon']
    odf['LATITUDE'] = odf['Lat']
    odf['ozone'] = odf['Conc']
    eds['O3'] = eds['O3_MDA8']

    # 插值与计算偏差
    odf['x'], odf['y'] = proj(odf['LONGITUDE'], odf['LATITUDE'])
    odf['mod'] = eds['O3'].sel(ROW=odf['y'].to_xarray(), COL=odf['x'].to_xarray(), method='nearest')
    odf['bias'] = odf['mod'] - odf['ozone']
    odf['grad_evna'] = odf['ozone'] / odf['mod']

    # 预测插值
    pdf = eds[['ROW', 'COL']].to_dataframe().reset_index()
    nn = nna_methods.NNA(method='voronoi', k=30)
    nn.fit(odf[['x', 'y']], odf[['ozone', 'mod', 'bias', 'grad_evna']])
    zdf = nn.predict(pdf[['COL', 'ROW']])
    pdf['vna_ozone'], pdf['vna_mod'], pdf['vna_bias'], pdf['vna_grad'] = zdf.T
    pds = pdf.set_index(['ROW', 'COL']).to_xarray()
    pds['avna_ozone'] = eds['O3'] - pds['vna_bias']

    #eVNA=O3*Σ(VNAobs/VNAmod) ieVNA=O3*(ΣVNAobs/ΣVNAmod)    
    pds['ievna_ozone'] = eds['O3']*pds['vna_ozone']/pds['vna_mod']
    pds['evna_ozone'] = eds['O3']*pds['vna_grad']

    #添加模型数据
    pds['O3'] = eds['O3']

    # 保存数据
    pdf = pds.to_dataframe().reset_index()
    pdf['longitude'], pdf['latitude'] = proj(pdf['COL'], pdf['ROW'], inverse=True)
    output_data_path = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/SpatialData_{suffix}.csv'
    pdf[['ROW', 'COL', 'longitude', 'latitude','O3','vna_ozone','vna_mod','ievna_ozone','evna_ozone','avna_ozone']].to_csv(output_data_path, index=False)
    print(f"插值数据已保存到 {output_data_path}")

    # 保存图像
    output_image_path_vnaobs = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/equates_vnaobs_{suffix}.png'
    output_image_path_vnamod = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/equates_vnamod_{suffix}.png'
    output_image_path_avna = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/equates_avna_{suffix}.png'
    output_image_path_evna = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/equates_evna_{suffix}.png'
    output_image_path_ievna = f'D:/Project/DataFusion/runaveragedatafusion/python_output_{year}/equates_ievna_{suffix}.png'

    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    qm = eds['O3'].plot(ax=axx[0])
    qm = pds['vna_ozone'].plot(ax=axx[1], norm=qm.norm)
    qm = (pds['vna_ozone'] - eds['O3']).plot(ax=axx[2])
    axx[0].set(title='EQUATES')
    axx[1].set(title='VNA(obs)')
    axx[2].set(title='VNA(obs) - EQUATES')
    fig.savefig(output_image_path_vnaobs)

    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    qm = eds['O3'].plot(ax=axx[0])
    qm = pds['vna_mod'].plot(ax=axx[1], norm=qm.norm)
    qm = (pds['vna_mod'] - eds['O3']).plot(ax=axx[2])
    axx[0].set(title='EQUATES')
    axx[1].set(title='VNA(mod)')
    axx[2].set(title='VNA(mod) - EQUATES')
    fig.savefig(output_image_path_vnamod)

    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    qm = eds['O3'].plot(ax=axx[0])
    qm = pds['avna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
    qm = (pds['avna_ozone'] - eds['O3']).plot(ax=axx[2])
    axx[0].set(title='EQUATES')
    axx[1].set(title='aVNA')
    axx[2].set(title='aVNA(obs) - EQUATES')
    fig.savefig(output_image_path_avna)

    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    qm = eds['O3'].plot(ax=axx[0])
    qm = pds['ievna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
    qm = (pds['ievna_ozone'] - eds['O3']).plot(ax=axx[2])
    axx[0].set(title='EQUATES')
    axx[1].set(title='ieVNA')
    axx[2].set(title='ieVNA(obs) - EQUATES')
    fig.savefig(output_image_path_ievna)

    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    qm = eds['O3'].plot(ax=axx[0])
    qm = pds['evna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
    qm = (pds['evna_ozone'] - eds['O3']).plot(ax=axx[2])
    axx[0].set(title='EQUATES')
    axx[1].set(title='eVNA')
    axx[2].set(title='eVNA(obs) - EQUATES')
    fig.savefig(output_image_path_evna)



# 处理函数，先插值后均值
def process_data_daily(years, months_list):
    for year in years:
        for selected_months in months_list:
            print(f"正在处理年份: {year}, 月份: {selected_months}")
            odf, eds = load_data(year, selected_months)
            if odf is None or eds is None:
                print("数据加载失败，停止处理。")
                return

            suffix = generate_suffix(selected_months=selected_months, year=year)

            # 初始化存储每天插值结果的列表
            daily_results = []

            #test
            # 定义要处理的日期列表
            selected_dates = [pd.Timestamp('2017-09-01'), pd.Timestamp('2017-09-01')]

            # 获取所有唯一的日期并筛选出选定的日期
            unique_dates = eds['date'].unique()
            unique_dates = [date for date in unique_dates if date in selected_dates]
            #test

            # # 获取所有唯一的日期
            # unique_dates = eds['date'].unique()

            for date in unique_dates:
                # 筛选当天的数据
                eds_daily = eds[eds['date'] == date]
                odf_daily = odf[odf['date'] == date]

                # 分组计算
                Model_O3_MDA8_avg = eds_daily.groupby(['column', 'row']).agg({
                    'O3_MDA8': 'mean',#同一天多个值进行平均
                    'longitude': 'first',
                    'latitude': 'first',
                    'Lambert_X': 'first',
                    'LAMBERT_Y': 'first'
                }).reset_index()

                Monitor_O3_MDA8_avg = odf_daily.groupby(['Site']).agg({
                    'Conc': 'mean',#同一天多个值进行平均
                    'Lat': 'first',
                    'Lon': 'first'
                }).reset_index()

                # 重命名并转换为 xarray
                odf_daily = Monitor_O3_MDA8_avg
                eds_daily = Model_O3_MDA8_avg
                eds_daily = xr.Dataset.from_dataframe(Model_O3_MDA8_avg.set_index(['row', 'column']))
                eds_daily = eds_daily.rename({'row': 'ROW', 'column': 'COL'})
                odf_daily['LONGITUDE'] = odf_daily['Lon']
                odf_daily['LATITUDE'] = odf_daily['Lat']
                odf_daily['ozone'] = odf_daily['Conc']
                eds_daily['O3'] = eds_daily['O3_MDA8']

                # 插值与计算偏差
                odf_daily['x'], odf_daily['y'] = proj(odf_daily['LONGITUDE'], odf_daily['LATITUDE'])
                odf_daily['mod'] = eds_daily['O3'].sel(ROW=odf_daily['y'].to_xarray(), COL=odf_daily['x'].to_xarray(), method='nearest')
                odf_daily['bias'] = odf_daily['mod'] - odf_daily['ozone']

                odf_daily['grad_evna'] = odf_daily['ozone'] / odf_daily['mod']

                # 预测插值
                pdf = eds_daily[['ROW', 'COL']].to_dataframe().reset_index()
                nn = nna_methods.NNA(method='voronoi', k=30)
                nn.fit(odf_daily[['x', 'y']], odf_daily[['ozone', 'mod', 'bias','grad_evna']])
                zdf = nn.predict(pdf[['COL', 'ROW']])
                pdf['vna_ozone'], pdf['vna_mod'], pdf['vna_bias'], pdf['vna_grad'] = zdf.T
                pds = pdf.set_index(['ROW', 'COL']).to_xarray()
                pds['avna_ozone'] = eds_daily['O3'] - pds['vna_bias']
                
                #eVNA=O3*Σ(VNAobs/VNAmod) ieVNA=O3*(ΣVNAobs/ΣVNAmod)
                pds['ievna_ozone'] = eds_daily['O3']*pds['vna_ozone']/pds['vna_mod']
                pds['evna_ozone'] = eds_daily['O3']*pds['vna_grad']

                #进行插值后，将每天的模型数据O3也添加到pds
                pds['O3'] = eds_daily['O3']
                # 将每天的插值结果存储到列表中
                daily_results.append(pds)
                print(pds)

            # 对vna_ozone和avna_ozone、Model进行平均
            avg_vna_ozone = xr.concat([result['vna_ozone'] for result in daily_results], dim='date').mean(dim='date')
            avg_avna_ozone = xr.concat([result['avna_ozone'] for result in daily_results], dim='date').mean(dim='date')
            avg_vna_mod = xr.concat([result['vna_mod'] for result in daily_results], dim='date').mean(dim='date')
            avg_model_ozone = xr.concat([result['O3'] for result in daily_results], dim='date').mean(dim='date')
            avg_ieVNA_ozone = xr.concat([result['ievna_ozone'] for result in daily_results], dim='date').mean(dim='date')
            avg_eVNA_ozone = xr.concat([result['evna_ozone'] for result in daily_results], dim='date').mean(dim='date')

            # 创建一个新的Dataset来存储平均后的数据
            avg_pds = xr.Dataset({
                'vna_ozone': avg_vna_ozone,
                'avna_ozone': avg_avna_ozone,
                'vna_mod': avg_vna_mod,
                'O3': avg_model_ozone,
                'ievna_ozone': avg_ieVNA_ozone,
                'evna_ozone': avg_eVNA_ozone
            })

            # 保存数据
            pdf = avg_pds.to_dataframe().reset_index()
            pdf['longitude'], pdf['latitude'] = proj(pdf['COL'], pdf['ROW'], inverse=True)
            output_data_path = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/SpatialData_{suffix}.csv'
            #'longitude', 'latitude'#
            pdf[['ROW', 'COL','longitude','latitude','O3','vna_ozone','vna_mod','ievna_ozone','evna_ozone','avna_ozone']].to_csv(output_data_path, index=False)
            print(f"插值数据已保存到 {output_data_path}")

            # 图像路径
            output_image_path_vnaobs = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/equates_vnaobs_{suffix}.png'
            output_image_path_vnamod = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/equates_vnamod_{suffix}.png'
            output_image_path_avna = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/equates_avna_{suffix}.png'
            output_image_path_ievna = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/equates_ievna_{suffix}.png'
            output_image_path_evna = f'D:/Project/DataFusion/rundailydatafusion/python_output_{year}/equates_evna_{suffix}.png'

            #输出VNAobs、VNAmod、aVNA、ieVNA的图像
            fig, axx = plt.subplots(1, 3, figsize=(18, 4))
            qm = avg_pds['O3'].plot(ax=axx[0])
            qm = avg_pds['vna_ozone'].plot(ax=axx[1], norm=qm.norm)
            qm = (avg_pds['vna_ozone'] - avg_pds['O3']).plot(ax=axx[2])
            axx[0].set(title='EQUATES')
            axx[1].set(title='VNA(obs)')
            axx[2].set(title='VNA(obs) - EQUATES')
            fig.savefig(output_image_path_vnaobs)

            fig, axx = plt.subplots(1, 3, figsize=(18, 4))
            qm = avg_pds['O3'].plot(ax=axx[0])
            qm = avg_pds['vna_mod'].plot(ax=axx[1], norm=qm.norm)
            qm = (avg_pds['vna_mod'] - avg_pds['O3']).plot(ax=axx[2])
            axx[0].set(title='EQUATES')
            axx[1].set(title='VNA(mod)')
            axx[2].set(title='VNA(mod) - EQUATES')
            fig.savefig(output_image_path_vnamod)

            fig, axx = plt.subplots(1, 3, figsize=(18, 4))
            qm = avg_pds['O3'].plot(ax=axx[0])
            qm = avg_pds['avna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
            qm = (avg_pds['avna_ozone'] - avg_pds['O3']).plot(ax=axx[2])
            axx[0].set(title='EQUATES')
            axx[1].set(title='aVNA')
            axx[2].set(title='aVNA(obs) - EQUATES')
            fig.savefig(output_image_path_avna)

            fig, axx = plt.subplots(1, 3, figsize=(18, 4))
            qm = avg_pds['O3'].plot(ax=axx[0])
            qm = avg_pds['ievna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
            qm = (avg_pds['ievna_ozone'] - avg_pds['O3']).plot(ax=axx[2])
            axx[0].set(title='EQUATES')
            axx[1].set(title='ieVNA')
            axx[2].set(title='ieVNA(obs) - EQUATES')
            fig.savefig(output_image_path_ievna)

            fig, axx = plt.subplots(1, 3, figsize=(18, 4))
            qm = avg_pds['O3'].plot(ax=axx[0])
            qm = avg_pds['evna_ozone'].plot(ax=axx[1], norm=qm.norm, cmap='viridis')
            qm = (avg_pds['evna_ozone'] - avg_pds['O3']).plot(ax=axx[2])
            axx[0].set(title='EQUATES')
            axx[1].set(title='eVNA')
            axx[2].set(title='eVNA(obs) - EQUATES')
            fig.savefig(output_image_path_evna)

#先均值后插值
process_data_average(years=[2017], months_list=[[9]])

#先插值后均值
process_data_daily(years=[2017], months_list=[[9]])