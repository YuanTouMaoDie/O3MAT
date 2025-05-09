import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress

# -------------------- 数据加载 --------------------
data_fusion_dir = "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_average_data_fusion"

# Quarter 文件路径
quarter_files = {
    "Quarter1": os.path.join(data_fusion_dir, "2011_dailyIn_Quarter1Out_O3_Ozone_Data.csv"),
    "Quarter2": os.path.join(data_fusion_dir, "2011_dailyIn_Quarter2Out_O3_Ozone_Data.csv"),
    "Quarter3": os.path.join(data_fusion_dir, "2011_dailyIn_Quarter3Out_O3_Ozone_Data.csv"),
    "Quarter4": os.path.join(data_fusion_dir, "2011_dailyIn_Quarter4Out_O3_Ozone_Data.csv")
}

# Y 轴数据文件（包含 Period 列）
y_data_files = [
    "/DeepLearning/mnt/shixiansheng/data_fusion/output/seasonal_fused_data_2011.csv"
]

# -------------------- 创建文件夹 --------------------
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/scatter_plots_Python_vs_DFT_Season'
os.makedirs(output_dir, exist_ok=True)

# -------------------- Quarter 文件与季节的映射 --------------------
quarter_to_season = {
    "Quarter1": "JFM",
    "Quarter2": "AMJ",
    "Quarter3": "JAS",
    "Quarter4": "OND"
}

# -------------------- 定义绘图函数 --------------------
def plot_density_scatter(x_data, y_data, period_value, output_dir, x_label, y_label):
    """
    绘制散点密度图：x_data vs y_data。
    文件名包含对应的 Period 字段。
    """
    # 提取年份
    year = "2011"
    formatted_period = f"{year}_{period_value}"

    # -------------------- 密度散点图 --------------------
    xy = np.vstack([x_data, y_data])
    kde = gaussian_kde(xy)
    z = kde(xy)
    z = (z - z.min()) / (z.max() - z.min())  # 归一化

    # 绘制散点密度图
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_data, y_data, c=z, cmap='jet', s=20, alpha=0.8)
    fig.colorbar(scatter, ax=ax)

    # 添加 1:1 参考线
    max_val = max(x_data.max(), y_data.max())
    ax.plot([0, max_val], [0, max_val], 'k-', lw=0.5, label="1:1 line")

    # 添加回归线
    slope, intercept, r_value, _, _ = linregress(x_data, y_data)
    regression_line = slope * np.array([0, max_val]) + intercept
    ax.plot([0, max_val], regression_line, 'r-', lw=0.5, label="Regression Line")
    r_squared = r_value ** 2
    mae = np.mean(np.abs(y_data - x_data))  # MAE 计算公式
    rmse = np.sqrt(np.mean((y_data - x_data) ** 2))  # RMSE 计算公式
    ax.text(0.95, 0.05, f"$R^2$ = {r_squared:.4f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # 设置标题和标签
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title(f'{formatted_period} - {y_label} vs {x_label}', fontsize=13)

    # 保存图像
    output_file_name = f'{formatted_period}_{y_label}_vs_{x_label}_density.png'
    output_path = os.path.join(output_dir, output_file_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.close()
    print(f"散点图已保存至 {output_path}")

# -------------------- 转置并排序 ROW 和 COL --------------------
def transpose_and_sort_data(y_data):
    """
    转置并排序 y_data 中的 ROW 和 COL 数据。
    """
    # 转置 ROW 和 COL
    y_data['ROW'], y_data['COL'] = y_data['COL'], y_data['ROW']

    # 按 ROW 和 COL 排序
    y_data = y_data.sort_values(by=['ROW', 'COL']).reset_index(drop=True)
    return y_data

# -------------------- 读取和处理数据 --------------------
def process_data(quarter_data, y_data, output_dir, file_name, season):
    """
    处理单个 y_data 文件。
    """
    # 转置并排序 y_data 中的 ROW 和 COL
    # print(f"正在处理文件: {file_name}")
    # y_data = transpose_and_sort_data(y_data)

    # 检查 y_data 的 Period 列是否包含关键词
    y_subset = y_data[y_data['Period'].str.contains(season, case=False, na=False)]
    
    # 如果没有匹配的数据，跳过
    if y_subset.empty:
        print(f"警告：Period 列中未找到关键词 '{season}'，跳过处理。")
        return

    # 提取对应 Period 的 y_data
    y_vna = y_subset['vna_ozone'].values  # y 轴目标列：vna_ozone
    y_evna = y_subset['evna_ozone'].values  # y 轴目标列：evna_ozone

    # 确保数据长度一致（按行顺序匹配）
    if len(y_vna) != len(quarter_data) or len(y_evna) != len(quarter_data):
        print(f"错误：数据长度不一致（季节={season}），跳过处理。")
        return

    # 提取对应 Period 的 x_data
    x_vna = quarter_data['O3_VNA'].values  # x 轴目标列：O3_VNA
    x_evna = quarter_data['O3_eVNA'].values  # x 轴目标列：O3_eVNA

    # 打印正在处理的信息
    print(f"正在处理 {season} 数据：")
    print(f"- Python_VNA vs. DFT_VNA: 使用列 'O3_VNA'")
    print(f"- Python_eVNA vs. DFT_eVNA: 使用列 'O3_eVNA'")

    # 绘制 Python_VNA vs. DFT_VNA
    plot_density_scatter(
        x_data=x_vna,
        y_data=y_vna,
        period_value=season,
        output_dir=output_dir,
        x_label='DFT_VNA',
        y_label='Python_VNA'
    )

    # 绘制 Python_eVNA vs. DFT_eVNA
    plot_density_scatter(
        x_data=x_evna,
        y_data=y_evna,
        period_value=season,
        output_dir=output_dir,
        x_label='DFT_eVNA',
        y_label='Python_eVNA'
    )

# -------------------- 执行数据处理 --------------------
if __name__ == "__main__":
    # 加载 y_data 文件
    print("正在加载 y_data 文件...")
    y_data_list = [pd.read_csv(y_data_file) for y_data_file in y_data_files]
    print("y_data 文件加载完成。")

    # 处理每个 Quarter 文件
    for quarter_name, quarter_file in quarter_files.items():
        print(f"正在加载 Quarter 文件: {quarter_file}")
        # 跳过第一行数据
        quarter_data = pd.read_csv(quarter_file, skiprows=1)
        season = quarter_to_season.get(quarter_name, "Unknown")
        print(f"Quarter 文件对应的季节: {season}")

        # 处理每个 y_data 文件
        for y_data in y_data_list:
            file_name = os.path.basename(quarter_file).split(".")[0]  # 提取文件名（不含扩展名）
            process_data(quarter_data, y_data, output_dir, file_name, season)

    print("所有文件处理完成。")