import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
import re

# -------------------- 数据加载 --------------------
fusion_output_files = [
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/2011_dailyIn_AnnualOut_O3_Ozone_Data.csv",
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/2011_dailyIn_Quarter1Out_O3_Ozone_Data.csv",
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/2011_dailyIn_Quarter2Out_O3_Ozone_Data.csv",
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/2011_dailyIn_Quarter3Out_O3_Ozone_Data.csv",
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/2011_dailyIn_Quarter4Out_O3_Ozone_Data.csv",
    "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion/summer.csv"
]

# -------------------- 创建文件夹 --------------------
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/scatter_plots_DFT_FirstAverage'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# -------------------- 定义绘图函数 --------------------
def plot_density_scatter(dataframe, model_column, vna_ozone_column, evna_ozone_column, output_dir, period_value, file_name):
    """
    绘制两张散点密度图：一张为 model vs vna_ozone，另一张为 model vs evna_ozone。
    文件名包含对应的 Period 字段。
    """
    # 获取数据
    x_model = dataframe[model_column]
    y_vna = dataframe[vna_ozone_column]
    y_evna = dataframe[evna_ozone_column]

    # 获取文件名的基名
    file_base_name = os.path.basename(file_name).split(".")[0]
    print(f"处理文件: {file_base_name}")

    # 根据文件名判断标题前缀
    if 'processed' in file_base_name:
        title_prefix = 'Daily'
    elif 'dailyIn' in file_base_name:
        title_prefix = 'Seasonal'
    else:
        title_prefix = 'Unknown'  # 如果没有匹配的前缀
    
    print(f"title_prefix: {title_prefix}")  # 打印title_prefix

    # 提取年份
    year = re.search(r"\d{4}", file_base_name).group(0)  # 提取年份
    print(f"年份: {year}")

    # 根据文件名提取周期
    if 'Annual' in file_base_name:
        period_value = 'Annual'
    elif 'Quarter1' in file_base_name:
        period_value = 'JFM'
    elif 'Quarter2' in file_base_name:
        period_value = 'AMJ'
    elif 'Quarter3' in file_base_name:
        period_value = 'JAS'
    elif 'Quarter4' in file_base_name:
        period_value = 'OND'
    elif 'summer' in file_base_name:
        period_value = 'Apr_Sep'

    print(f"周期: {period_value}")

    # 拼接成期望的格式，如：2011_JAS
    formatted_period = f"{year}_{period_value}"
    print(f"生成的期望格式: {formatted_period}")

    # -------------------- vna_ozone (O3_VNA) 密度散点图 --------------------
    xy_vna = np.vstack([x_model, y_vna])
    kde_vna = gaussian_kde(xy_vna)
    z_vna = kde_vna(xy_vna)
    z_vna = (z_vna - z_vna.min()) / (z_vna.max() - z_vna.min())  # 归一化

    # 绘制散点密度图
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_model, y_vna, c=z_vna, cmap='jet', s=20, alpha=0.8)
    fig.colorbar(scatter, ax=ax)

    # 添加 1:1 参考线
    max_val_vna = max(x_model.max(), y_vna.max())
    ax.plot([0, max_val_vna], [0, max_val_vna], 'k-', lw=0.5, label="1:1 line")

    # 添加回归线
    slope_vna, intercept_vna, r_value_vna, _, _ = linregress(x_model, y_vna)
    regression_line_vna = slope_vna * np.array([0, max_val_vna]) + intercept_vna
    ax.plot([0, max_val_vna], regression_line_vna, 'r-', lw=0.5, label="Regression Line")
    r_squared_vna = r_value_vna ** 2
    mae_vna = np.mean((y_vna - x_model))
    rmse_vna = np.sqrt(np.mean((y_vna - x_model) ** 2))
    ax.text(0.95, 0.05, f"$R^2$ = {r_squared_vna:.4f}\nMAE = {mae_vna:.3f}\nRMSE = {rmse_vna:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # 设置标题和标签
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('O3_VNA', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    # 将标题放置到图像顶部
    fig.subplots_adjust(top=0.85)
    ax.set_title(f'{title_prefix} {formatted_period} - O3_VNA vs Model', fontsize=13, loc='center')

    # 保存图像
    output_file_name = f'{file_base_name}_{formatted_period}_O3_VNA_density.png'
    vna_output_path = os.path.join(output_dir, output_file_name)
    plt.tight_layout()
    plt.savefig(vna_output_path, dpi=400)
    plt.close()
    print(f"O3_VNA 散点密度图已保存至 {vna_output_path}")

    # -------------------- evna_ozone (O3_eVNA) 密度散点图 --------------------
    xy_evna = np.vstack([x_model, y_evna])
    kde_evna = gaussian_kde(xy_evna)
    z_evna = kde_evna(xy_evna)
    z_evna = (z_evna - z_evna.min()) / (z_evna.max() - z_evna.min())  # 归一化

    # 绘制散点密度图
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_model, y_evna, c=z_evna, cmap='jet', s=20, alpha=0.8)
    fig.colorbar(scatter, ax=ax)

    # 添加 1:1 参考线
    max_val_evna = max(x_model.max(), y_evna.max())
    ax.plot([0, max_val_evna], [0, max_val_evna], 'k-', lw=0.5, label="1:1 line")

    # 添加回归线
    slope_evna, intercept_evna, r_value_evna, _, _ = linregress(x_model, y_evna)
    regression_line_evna = slope_evna * np.array([0, max_val_evna]) + intercept_evna
    ax.plot([0, max_val_evna], regression_line_evna, 'r-', lw=0.5, label="Regression Line")
    r_squared_evna = r_value_evna ** 2
    mae_evna = np.mean((y_evna - x_model))
    rmse_evna = np.sqrt(np.mean((y_evna - x_model) ** 2))
    ax.text(0.95, 0.05, f"$R^2$ = {r_squared_evna:.4f}\nMAE = {mae_evna:.3f}\nRMSE = {rmse_evna:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # 设置标题和标签
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('O3_eVNA', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    # 将标题放置到图像顶部
    fig.subplots_adjust(top=0.85)
    ax.set_title(f'{title_prefix} {formatted_period} - O3_eVNA vs Model', fontsize=13, loc='center')

    # 保存图像
    output_file_name = f'{file_base_name}_{formatted_period}_O3_eVNA_density.png'
    evna_output_path = os.path.join(output_dir, output_file_name)
    plt.tight_layout()
    plt.savefig(evna_output_path, dpi=400)
    plt.close()
    print(f"O3_eVNA 散点密度图已保存至 {evna_output_path}")

# -------------------- 读取和处理多个文件 --------------------
def process_file(fusion_output_file):
    # 读取数据
    if not os.path.exists(fusion_output_file):
        print(f"文件不存在: {fusion_output_file}")
        return
    
    print(f"开始处理文件: {fusion_output_file}")
    df_data = pd.read_csv(fusion_output_file, skiprows=1)  # 跳过第一行
    print(f"文件 {fusion_output_file} 已成功加载，数据行数: {df_data.shape[0]}")

    # 提取文件名信息
    file_base_name = os.path.basename(fusion_output_file).split(".")[0]

    # 提取年份
    year = re.search(r"\d{4}", file_base_name).group(0)  # 提取年份
    print(f"年份: {year}")

    # 提取周期
    if 'Annual' in file_base_name:
        period_value = 'Annual'
    elif 'Quarter1' in file_base_name:
        period_value = 'JFM'
    elif 'Quarter2' in file_base_name:
        period_value = 'AMJ'
    elif 'Quarter3' in file_base_name:
        period_value = 'JAS'
    elif 'Quarter4' in file_base_name:
        period_value = 'OND'
    elif 'summer' in file_base_name:
        period_value = 'Apr_Sep'

    # 绘制图像
    print(f"开始绘制周期: {period_value}")
    plot_density_scatter(df_data, 'O3_Model', 'O3_VNA', 'O3_eVNA', output_dir, period_value, fusion_output_file)

# -------------------- 逐个文件处理 --------------------
for file in fusion_output_files:
    process_file(file)
