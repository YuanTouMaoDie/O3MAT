import pandas as pd
import numpy as np
from tqdm import tqdm
from dask import delayed, compute
import numba
import calendar
import matplotlib.pyplot as plt


# 定义需要读取的列
usecols = ['site_id', 'dateon', 'O3']

# 定义每列的数据类型，减少内存使用
dtype = {
    'site_id': 'object',
    'dateon': 'object',
    'O3': 'float32'
}


@numba.jit(nopython=True)
def calculate_weighted_values(values):
    return values / (1 + 4403 * np.exp(-126 * values))


@numba.jit(nopython=True)
def sliding_window_sum(arr, window_size):
    n = len(arr)
    result = np.empty(n - window_size + 1)
    for i in range(n - window_size + 1):
        result[i] = np.sum(arr[i:i + window_size])
    return result


@delayed
def calculate_single_w126(group):
    values = group['O3'].values
    months = group['Month'].values
    years = group['Year'].values
    hours = group['hour'].values

    # 筛选白天数据
    daytime_mask = (hours >= 8) & (hours < 20)
    daytime_values = values[daytime_mask]
    daytime_months = months[daytime_mask]
    daytime_years = years[daytime_mask]

    # 按年和月分组并计算每月的加权和，同时处理缺失数据
    unique_years_months, idx = np.unique(list(zip(daytime_years, daytime_months)), axis=0, return_index=True)
    monthly_weighted_sum = np.zeros(len(unique_years_months))
    for i, (year_month, start_idx) in enumerate(zip(unique_years_months, idx)):
        year, month = year_month
        mask = (daytime_years == year) & (daytime_months == month)
        monthly_values = daytime_values[mask]

        # 计算该月的实际天数
        _, num_days = calendar.monthrange(year, month)
        total_possible_hours = 12 * num_days  # 每天 12 个白天小时
        available_hours = len(monthly_values)
        available_ratio = available_hours / total_possible_hours

        # 如果可用比例小于 75%，跳过该月
        if available_ratio < 0.75:
            monthly_weighted_sum[i] = np.nan
            continue

        # 计算加权值
        weighted_values = calculate_weighted_values(monthly_values)

        # 调整加权和以处理缺失数据
        adjustment_factor = 1 / available_ratio
        monthly_weighted_sum[i] = np.sum(weighted_values) * adjustment_factor

    # 定义特定的 3 个月组合
    specific_month_combinations = [(3, 4, 5), (4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10)]
    three_month_sums = []
    for start_month, middle_month, end_month in specific_month_combinations:
        start_mask = (unique_years_months[:, 1] == start_month)
        middle_mask = (unique_years_months[:, 1] == middle_month)
        end_mask = (unique_years_months[:, 1] == end_month)

        # 检查每个月是否有有效的数据
        if np.sum(start_mask) > 0 and np.sum(middle_mask) > 0 and np.sum(end_mask) > 0:
            start_idx = np.where(start_mask)[0][0]
            middle_idx = np.where(middle_mask)[0][0]
            end_idx = np.where(end_mask)[0][0]

            # 检查每个月的值是否为有效（非 NaN）
            if not np.isnan(monthly_weighted_sum[start_idx]) and not np.isnan(monthly_weighted_sum[middle_idx]) and not np.isnan(monthly_weighted_sum[end_idx]):
                three_month_sum = monthly_weighted_sum[start_idx] + monthly_weighted_sum[middle_idx] + monthly_weighted_sum[end_idx]
                three_month_sums.append(three_month_sum)

    if len(three_month_sums) > 0:
        return np.max(three_month_sums)
    return np.nan


@delayed
def calculate_w126_metric(df_data, save_path, project_name):
    print("开始计算 W126 指标...")
    # 转换单位，ppbv to ppm
    df_data['O3'] = df_data['O3'] / 1000
    df_data['dateon'] = pd.to_datetime(df_data['dateon'])
    df_data['Year'] = df_data['dateon'].dt.year
    df_data['Month'] = df_data['dateon'].dt.month
    df_data['hour'] = df_data['dateon'].dt.hour

    all_w126_metrics = []
    grouped = df_data.groupby(['site_id'])
    for site_id, group in tqdm(grouped, desc="Processing sites"):
        task = calculate_single_w126(group)
        result = compute(task)[0]
        w126_metrics = {'site_id': site_id, 'Period': 'W126', 'O3': result}
        all_w126_metrics.append(w126_metrics)

    w126_df = pd.DataFrame(all_w126_metrics)
    # 按照 O3 列的值从大到小排序
    w126_df = w126_df.sort_values(by='O3', ascending=False)
    w126_output_file = f'{save_path}/{project_name}_W126.csv'
    w126_df[['site_id', 'O3', 'Period']].to_csv(w126_output_file, index=False)
    print(f"W126 指标数据已保存到: {w126_output_file}")
    print("W126 指标计算完成.")
    return w126_df


def save_w126_metrics(save_path, project_name, file_path):
    print("开始保存 W126 指标...")

    # 使用 dask 直接读取 CSV 文件
    dask_df = pd.read_csv(file_path, usecols=usecols, dtype=dtype)

    print("文件读取完毕。")

    # 延迟计算
    w126_task = calculate_w126_metric(dask_df, save_path, project_name)

    # 并行计算 W126
    w126_df = compute(w126_task)[0]

    print("W126 指标保存完成.")
    return w126_df


def plot_cumulative_distribution(w126_df):
    # 去除 O3 列中的 NaN 值
    w126_values = w126_df['O3'].dropna()
    # 对 W126 值进行排序
    sorted_w126 = np.sort(w126_values)
    # 计算累计百分比
    cumulative_percentage = np.arange(1, len(sorted_w126) + 1) / len(sorted_w126)

    # 绘制累计分布图
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_w126, cumulative_percentage * 100)
    plt.xlabel('W126 数值')
    plt.ylabel('累计百分比 (%)')
    plt.title('W126 数值累计分布')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("开始读取输入文件...")
    # 读取输入文件
    file_path = "/backupdata/data_EPA/aq_obs/routine/2010/AQS_hourly_data_2010_LatLon.csv"

    # 定义保存路径和项目名称
    save_path = r"/DeepLearning/mnt/shixiansheng/data_fusion/output/W126"
    project_name = "2010_Monitor"

    # 调用函数计算指标并保存结果
    w126_df = save_w126_metrics(save_path, project_name, file_path)

    # 绘制累计分布图
    plot_cumulative_distribution(w126_df)

    print(f'W126 指标文件已保存到: {save_path}/{project_name}_W126.csv')
    