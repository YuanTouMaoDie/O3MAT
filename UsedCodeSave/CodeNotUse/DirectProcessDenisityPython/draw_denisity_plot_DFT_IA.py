import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress

# -------------------- 数据加载 --------------------
data_fusion_dir = "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_daily_data_fusion"
model_data_file = os.path.join(data_fusion_dir, "processed_results_O3_Model.csv")
vna_data_file = os.path.join(data_fusion_dir, "processed_results_O3_VNA.csv")
evna_data_file = os.path.join(data_fusion_dir, "processed_results_O3_eVNA.csv")

# -------------------- 创建文件夹 --------------------
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/scatter_plots_DFT_Firstinterpolate'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# -------------------- 定义绘图函数 --------------------
def plot_density_scatter(model_data, vna_data, evna_data, period_column, output_dir, period_value):
    """
    绘制两张散点密度图：一张为 model vs vna_ozone，另一张为 model vs evna_ozone。
    文件名包含对应的 Period 字段。
    """
    # 获取数据
    x_model = model_data[period_column]
    y_vna = vna_data[period_column]
    y_evna = evna_data[period_column]

    # 提取年份
    year = "2011"  # 由于没有时间信息，日期全定为2011年

    # 根据 period_column 判断标题前缀
    if period_column == 'annual_mean':
        period_value = 'Annual'
    elif period_column == 'JFM_mean':
        period_value = 'JFM'
    elif period_column == 'AMJ_mean':
        period_value = 'AMJ'
    elif period_column == 'JAS_mean':
        period_value = 'JAS'
    elif period_column == 'OND_mean':
        period_value = 'OND'
    elif period_column == 'summer_mean':
        period_value = 'Apr_Sep'

    # 拼接成期望的格式，如：2011_JAS
    formatted_period = f"{period_value}_{year}"
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
    ax.set_title(f'{formatted_period} - O3_VNA vs Model', fontsize=13, loc='center')

    # 保存图像
    output_file_name = f'{formatted_period}_O3_VNA_density.png'
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
    ax.set_title(f'{formatted_period} - O3_eVNA vs Model', fontsize=13, loc='center')

    # 保存图像
    output_file_name = f'{formatted_period}_O3_eVNA_density.png'
    evna_output_path = os.path.join(output_dir, output_file_name)
    plt.tight_layout()
    plt.savefig(evna_output_path, dpi=400)
    plt.close()
    print(f"O3_eVNA 散点密度图已保存至 {evna_output_path}")

# -------------------- 读取和处理数据 --------------------
def process_data(model_data_file, vna_data_file, evna_data_file, output_dir):
    # 读取数据
    model_data = pd.read_csv(model_data_file)
    vna_data = pd.read_csv(vna_data_file)
    evna_data = pd.read_csv(evna_data_file)

    # 定义需要处理的周期列
    # period_columns-1 = ['annual_mean', 'JFM_mean', 'AMJ_mean', 'JAS_mean', 'OND_mean', 'summer_mean']
    period_columns = ['top_10_mean', '98th_percentile']
    # 逐个周期处理
    for period_column in period_columns:
        print(f"开始处理周期: {period_column}")
        plot_density_scatter(model_data, vna_data, evna_data, period_column, output_dir, period_column)

# -------------------- 执行数据处理 --------------------
process_data(model_data_file, vna_data_file, evna_data_file, output_dir)