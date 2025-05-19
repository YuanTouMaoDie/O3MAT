import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import numpy as np

# 设置字体为新罗马
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取气候区域映射文件
region_map = pd.read_csv('/DeepLearning/mnt/shixiansheng/data_fusion/output/Region/2011_299*459_9ClimateRegions.csv')
region_map = region_map[['ROW', 'COL', 'ClimateRegion']].dropna()
region_map['ClimateRegion'] = region_map['ClimateRegion'].astype(int)
region_map = region_map[region_map['ClimateRegion'].between(0, 8)]  # 仅保留0-8的气候区域

# 定义NOAA气候区域映射 (RegionCode -> CMAQName)
region_mapping = {
    '001': {'id': 0, 'name': 'Northeast', 'cmaq_name': 'NE_CR'},
    '002': {'id': 1, 'name': 'North Rockies and Plains', 'cmaq_name': 'NRP_CR'},
    '003': {'id': 2, 'name': 'Northwest', 'cmaq_name': 'NW_CR'},
    '004': {'id': 3, 'name': 'Ohio Valley', 'cmaq_name': 'CEN_CR'},
    '005': {'id': 4, 'name': 'South', 'cmaq_name': 'S_CR'},
    '006': {'id': 5, 'name': 'Southeast', 'cmaq_name': 'SE_CR'},
    '007': {'id': 6, 'name': 'Southwest', 'cmaq_name': 'SW_CR'},
    '008': {'id': 7, 'name': 'Upper Midwest', 'cmaq_name': 'UPMW_CR'},
    '009': {'id': 8, 'name': 'West', 'cmaq_name': 'W_CR'},
    'USA': {'id': 9, 'name': 'USA', 'cmaq_name': 'USA'}
}

# 定义需要绘图的指标
metrics = ['model', 'vna_ozone', 'evna_ozone', 'avna_ozone', 'ds_ozone', 'harvard_ml']

# 变量名映射（替换为更友好的显示名称）
method_display_names = {
    'vna_ozone': 'VNA',
    'evna_ozone': 'eVNA',
    'avna_ozone': 'aVNA',
    'ds_ozone': 'Downscaler',
    'harvard_ml': 'Harvard ML',
    'model': 'EQUATES'
}

# 定义Periods（可自定义）
periods = ['DJF', 'MAM', 'JJA', 'SON', 'Annual', 'Apr-Sep']
# periods = ['top-10']  # 示例：只处理特定时段

# 读取原始数据
years = list(range(2002, 2020))  # 2002-2019共18年
all_data = pd.DataFrame()

print("正在读取每年的数据...")
for year in tqdm(years):
    file_path = f'/DeepLearning/mnt/shixiansheng/data_fusion/output/DailyData_WithoutCV/{year}_Data_WithoutCV_Metrics.csv'
    try:
        data = pd.read_csv(file_path)
        data['Year'] = year
        # 合并气候区域信息
        data = data.merge(region_map, on=['ROW', 'COL'], how='left')
        all_data = pd.concat([all_data, data], ignore_index=True)
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件。")

# 创建输出目录
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/9ClimateRegion_HeatMapForTimeseries'
os.makedirs(output_dir, exist_ok=True)

### 计算全局图例范围（排除top-10时段）
global_min = float('inf')
global_max = float('-inf')

print("正在计算全局图例范围（非top-10时段）...")
for period in periods:
    if period == 'top-10':
        continue  # 跳过top-10，单独计算其范围
    
    period_data = all_data[all_data['Period'] == period].copy()
    if period_data.empty:
        continue
    
    for region_code, region_info in region_mapping.items():
        region_id = region_info['id']
        if region_id == 9:
            region_period_data = period_data.copy()
        else:
            region_period_data = period_data[period_data['ClimateRegion'] == region_id].copy()
        
        if region_period_data.empty:
            continue
        
        for metric in metrics:
            if metric in region_period_data.columns:
                # 处理harvard_ml的年份限制
                if metric == 'harvard_ml':
                    yearly_mean = region_period_data[region_period_data['Year'] <= 2016]\
                                .groupby('Year')[metric].mean()
                else:
                    yearly_mean = region_period_data.groupby('Year')[metric].mean()
                
                if not yearly_mean.empty:
                    # 重新索引以匹配完整年份范围，缺失值用NaN填充
                    yearly_mean = yearly_mean.reindex(years)
                    current_min = yearly_mean.min()
                    current_max = yearly_mean.max()
                    if not np.isnan(current_min):  # 忽略全NaN的情况
                        global_min = min(global_min, current_min)
                    if not np.isnan(current_max):
                        global_max = max(global_max, current_max)

### 计算top-10时段的图例范围
top10_min = float('inf')
top10_max = float('-inf')

print("正在计算top-10时段图例范围...")
for period in [p for p in periods if p == 'top-10']:
    period_data = all_data[all_data['Period'] == period].copy()
    if period_data.empty:
        continue
    
    for region_code, region_info in region_mapping.items():
        region_id = region_info['id']
        if region_id == 9:
            region_period_data = period_data.copy()
        else:
            region_period_data = period_data[period_data['ClimateRegion'] == region_id].copy()
        
        if region_period_data.empty:
            continue
        
        for metric in metrics:
            if metric in region_period_data.columns:
                # 处理harvard_ml的年份限制
                if metric == 'harvard_ml':
                    yearly_mean = region_period_data[region_period_data['Year'] <= 2016]\
                                .groupby('Year')[metric].mean()
                else:
                    yearly_mean = region_period_data.groupby('Year')[metric].mean()
                
                if not yearly_mean.empty:
                    yearly_mean = yearly_mean.reindex(years)
                    current_min = yearly_mean.min()
                    current_max = yearly_mean.max()
                    if not np.isnan(current_min):
                        top10_min = min(top10_min, current_min)
                    if not np.isnan(current_max):
                        top10_max = max(top10_max, current_max)

# 为每个气候区、指标和Period生成热力图
print("正在生成热力图...")
for period in tqdm(periods, desc="处理时间段"):
    # 按Period筛选数据
    period_data = all_data[all_data['Period'] == period].copy()
    
    if period_data.empty:
        print(f"警告: 时间段 {period} 没有数据，跳过绘图")
        continue
    
    # 确定当前Period的图例范围
    if period == 'top-10':
        vmin, vmax = top10_min, top10_max
    else:
        vmin, vmax = global_min, global_max
    
    # 处理图例范围为空的情况（例如所有数据均为NaN）
    if np.isnan(vmin) or np.isnan(vmax):
        print(f"警告: {period} 时段图例范围无效，使用自动范围")
        vmin, vmax = None, None  # 让seaborn自动计算范围
    
    for region_code, region_info in tqdm(region_mapping.items(), desc="处理区域", leave=False):
        region_id = region_info['id']
        region_name = region_info['name']
        region_cmaq_name = region_info['cmaq_name']
        
        # 创建区域数据
        if region_id == 9:  # 全美国
            region_period_data = period_data.copy()
        else:
            region_period_data = period_data[period_data['ClimateRegion'] == region_id].copy()
        
        if region_period_data.empty:
            print(f"警告: 区域 {region_name} ({region_cmaq_name}) 在时间段 {period} 没有数据，跳过绘图")
            continue
        
        heatmap_data = {}
        for metric in metrics:
            if metric not in region_period_data.columns:
                continue  # 跳过不存在的指标
            
            # 处理harvard_ml的年份限制
            if metric == 'harvard_ml':
                method_data = region_period_data[region_period_data['Year'] <= 2016]
            else:
                method_data = region_period_data
            
            # 按年份分组并计算平均值，重新索引以匹配完整年份
            yearly_mean = method_data.groupby('Year')[metric].mean().reindex(years)
            heatmap_data[method_display_names[metric]] = yearly_mean.values  # 使用显示名称作为键
        
        if not heatmap_data:
            continue  # 跳过无数据的情况
        
        # 创建DataFrame并转置（方法为行，年份为列）
        heatmap_df = pd.DataFrame(heatmap_data, index=years).T  # 转置后方法为行，年份为列
        
        # 绘制热力图
        #选择黑色对比增加区分度，边缘更小
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            heatmap_df,
            cmap='viridis',
            annot=True,
            fmt=".1f",
            cbar=True,
            linecolor='black',# 设置边界颜色
            # square=True,  # 强制单元格为正方形
            linewidths=0.1,  # 减小单元格边界宽度
            vmin=vmin,
            vmax=vmax,
            mask=heatmap_df.isnull()  # 隐藏缺失值背景色
        )
        
        # 设置标题和坐标轴标签（保持原图名不变）
        plt.title(f'{period}: {region_cmaq_name} O₃ (2002 - 2019)', fontsize=16)
        # plt.xlabel('Year', fontsize=10)
        # plt.ylabel('Method', fontsize=10)
        
        # 调整刻度标签
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 添加数据缺失说明（恢复注释）
        # if 'Harvard ML' in heatmap_df.index:
        #     plt.text(
        #         0.98, 0.02,
        #         "Note: Harvard ML data only available until 2016",
        #         transform=plt.gca().transAxes,
        #         ha='right', va='bottom',
        #         bbox=dict(facecolor='white', alpha=0.8, pad=4)
        #     )
        
        # 自定义颜色条标签
        cbar = ax.collections[0].colorbar
        cbar.set_label('(ppbv)', fontsize=12)  # 补充O₃标识
        
        # 保存图表
        filename = f"{output_dir}/{period}_{region_cmaq_name}_Ozone_Heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

print("所有热力图生成完成！")