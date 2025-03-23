import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
import re

# -------------------- 数据加载 --------------------
fusion_output_files = [
    "/DeepLearning/mnt/shixiansheng/data_fusion/output/seasonal_fused_data_2011.csv",
]

# -------------------- 创建文件夹 --------------------
output_dir = '/DeepLearning/mnt/shixiansheng/data_fusion/output/scatter_plots_Python_AI'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# -------------------- 定义绘图函数 --------------------
def plot_density_scatter(dataframe, model_column, ozone_column, period_column, output_dir, period_value, file_name, ozone_type, x_range=(None, None)):
    """
    绘制散点密度图：model vs ozone_column（vna 或 avna）。
    文件名包含对应的 Period 字段。
    """
    # 获取数据
    df_period = dataframe[dataframe[period_column].str.contains(period_value)]  # 使用 str.contains 来匹配季节

    # 如果数据为空，跳过
    if df_period.empty:
        print(f"数据中没有有效数据，跳过 Period: {period_value} 的绘图。")
        return

    # 获取数据
    x_model = df_period[model_column]
    y_ozone = df_period[ozone_column]

    # 获取文件名的基名
    file_base_name = os.path.basename(file_name).split(".")[0]

    # 根据文件名判断标题前缀
    keywords = ['daily', 'ia', 'seasonal', 'ai']
    if any(keyword in file_base_name.lower() for keyword in keywords):
        if 'daily' in file_base_name.lower() or 'ia' in file_base_name.lower():
            title_prefix = 'Daily'
        elif 'seasonal' in file_base_name.lower() or 'ai' in file_base_name.lower():
            title_prefix = 'Seasonal'
    else:
        title_prefix = 'Unknown'  # 如果没有匹配的前缀

    # 提取年份
    year = re.search(r"\d{4}", file_base_name).group(0)  # 提取年份

    # 只提取季节部分
    period_search = re.search(r"(DJF|MAM|JJA|SON)", period_value)
    if period_search:
        period_value = period_search.group(0)  # 提取季节部分

    # 拼接成期望的格式，如：2011_JAS
    formatted_period = f"{period_value}_{year}"

    # -------------------- 密度散点图 --------------------
    print(f"正在处理 {ozone_type} 数据，Period: {formatted_period}")

    # 核密度计算
    xy = np.vstack([x_model, y_ozone])
    kde = gaussian_kde(xy)
    z = kde(xy)
    z = (z - z.min()) / (z.max() - z.min())  # 归一化

    # 绘制散点密度图
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_model, y_ozone, c=z, cmap='jet', s=20, alpha=0.8)
    fig.colorbar(scatter, ax=ax)  # 删除了 label 参数

    # 添加 1:1 参考线
    max_val = max(x_model.max(), y_ozone.max())
    ax.plot([0, max_val], [0, max_val], 'k-', lw=0.5, label="1:1 line")

    #设置x和y坐标范围已知
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    
    #  添加回归线
    slope, intercept, r_value, _, _ = linregress(x_model, y_ozone)
    regression_line = slope * np.array([0, max_val]) + intercept
    ax.plot([0, max_val], regression_line, 'r-', lw=0.5, label="Regression Line")
    r_squared = r_value ** 2
    mae = np.mean((y_ozone - x_model))
    rmse = np.sqrt(np.mean((y_ozone - x_model) ** 2))
    ax.text(0.95, 0.05, f"$R^2$ = {r_squared:.4f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12)

    # 设置标题和标签
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(ozone_type, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    # 设置x轴范围
    if x_range != (None, None):
        ax.set_xlim(x_range)

    # 将标题放置到图像顶部
    fig.subplots_adjust(top=0.85)  # 调整标题的位置
    ax.set_title(f'{title_prefix} {formatted_period} - {ozone_type} vs Model', fontsize=15, loc='center')

    # 保存图像，文件名包含 Period 字段和输入文件名（不含路径）
    output_file_name = f'{file_base_name}_{formatted_period}_{ozone_type}_density.png'
    output_path = os.path.join(output_dir, output_file_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"{ozone_type} 散点密度图已保存至 {output_path}")

# -------------------- 读取和处理多个文件 --------------------
def process_file(fusion_output_file, ozone_types, selected_periods=None, x_range=(None, None)):
    print(f"正在处理文件: {fusion_output_file}")
    # 读取数据
    df_data = pd.read_csv(fusion_output_file)

    # 提取Period列
    period_column = 'Period'  # Period列
    model_column = 'model'
    vna_ozone_column = 'vna_ozone'
    avna_ozone_column = 'avna_ozone'
    evna_ozone_colum = 'evna_ozone'

    # 如果用户未指定特定的 Period，使用所有 Period
    if selected_periods is None:
        selected_periods = df_data[period_column].unique()

    # 只保留季节部分（DJF, MAM, JJA, SON）
    valid_periods = ['DJF', 'MAM', 'JJA', 'SON']
    selected_periods = [period for period in selected_periods if period in valid_periods]

    # 遍历每个Period并绘制图形
    for period_value in selected_periods:
        for ozone_type in ozone_types:
            if ozone_type == 'vna':
                ozone_column = vna_ozone_column
            elif ozone_type == 'avna':
                ozone_column = avna_ozone_column
            elif ozone_type == 'evna':
                ozone_column = evna_ozone_colum
            else:
                raise ValueError(f"未知的 ozone_type: {ozone_type}")

            plot_density_scatter(df_data, model_column, ozone_column, period_column, output_dir, period_value, fusion_output_file, ozone_type, x_range)

# -------------------- 主函数 --------------------
def main():
    # 选择处理类型（vna 或 avna）
    ozone_types = ['avna']  # 默认同时处理 vna 和 avna
    
    # 自定义 x 轴范围和 Period
    x_range = (None, None)  # 默认不限制 x 轴范围
    selected_periods = ['MAM','JJA','SON']  # 可以自定义选择特定的 Period 进行绘图

    # 处理多个文件
    for file in fusion_output_files:
        process_file(file, ozone_types, selected_periods, x_range)

# 运行主函数
if __name__ == "__main__":
    main()
