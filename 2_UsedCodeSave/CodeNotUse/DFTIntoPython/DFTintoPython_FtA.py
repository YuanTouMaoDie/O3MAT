import os
import pandas as pd

def merge_data_files(model_file, vna_file, evna_file, year_prefix):
    # 读取模型数据（model 和 vna_ozone）
    df_model = pd.read_csv(model_file)

    # 读取VNA数据
    df_vna = pd.read_csv(vna_file)

    # 读取EVNA数据
    df_evna = pd.read_csv(evna_file)

    # 提取包含时间段的列（如 JFM_mean, JAS_mean 等）
    time_columns = [col for col in df_model.columns if 'mean' in col or '98th_percentile' in col]  # e.g., JFM_mean, AMJ_mean
    
    # 创建一个空的DataFrame来存储最终的合并结果
    merged_df = pd.DataFrame()

    for period in time_columns:
        # 处理每个时间段的数据

        if '98th_percentile' in period:
            period_name = "98th"
        else:
            period_name = period.replace('_mean', '')  # 去掉'_mean'部分

        if period_name == "annual":
           period_name = "Annual"

        if period_name == "summer":
           period_name = "Apr-Sep"

        if period_name == "top_10":
           period_name = "top-10"

        # 给Period列添加自定义年份前缀
        period_name = f"{year_prefix}_{period_name}"  # 使用传入的年份前缀

        # 获取模型数据
        model_data = df_model[period].dropna().reset_index(drop=True)
        
        # 获取VNA数据
        vna_data = df_vna[period].dropna().reset_index(drop=True)
        
        # 获取EVNA数据
        evna_data = df_evna[period].dropna().reset_index(drop=True)

        # 合并模型数据，VNA数据和EVNA数据
        temp_df = pd.DataFrame({
            'model': model_data,
            'vna_ozone': vna_data,
            'evna_ozone': evna_data,
            'Period': [period_name] * len(model_data)
        })

        # 获取每个Period的长度
        length = len(temp_df)

        # 计算COL和ROW的值
        rows_per_col = 299  # ROW从1到299
        cols = (length - 1) // rows_per_col + 1  # COL最大值（459）

        # 为每个Period生成ROW和COL
        rows = []
        cols_list = []
        for i in range(length):
            row = (i // rows_per_col) + 1  # COL从1开始，最大到459
            col = (i % rows_per_col) + 1  # ROW从1开始，最大到299
            cols_list.append(col)
            rows.append(row)

        # 将ROW和COL列添加到temp_df中
        temp_df['COL'] = cols_list
        temp_df['ROW'] = rows

        # 合并当前时间段的数据
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # 在合并后的数据中按照Period和ROW、COL排序
    merged_df = merged_df.sort_values(by=['Period', 'ROW', 'COL']).reset_index(drop=True)

    # 交换ROW和COL的数值
    merged_df['ROW'], merged_df['COL'] = merged_df['COL'], merged_df['ROW']

    # 重新按照ROW和COL排序
    merged_df = merged_df.sort_values(by=['Period','ROW', 'COL']).reset_index(drop=True)

    # 将COL和ROW列放到前面
    cols = ['ROW', 'COL', 'model', 'vna_ozone', 'evna_ozone', 'Period']
    merged_df = merged_df[cols]

    return merged_df

# 输入文件路径
data_fusion_dir = "/DeepLearning/mnt/Devin/data_fusion/data_fusion_result/run_daily_data_fusion"
model_file = os.path.join(data_fusion_dir, "processed_results_new_quarter_O3_Model.csv")
vna_data_file = os.path.join(data_fusion_dir, "processed_results_new_quarter_O3_VNA.csv")
evna_data_file = os.path.join(data_fusion_dir, "processed_results_new_quarter_O3_eVNA.csv")

# 自定义年份前缀
year_prefix = "2011"  # 可以修改为其他年份，比如 "2012" 等

# 调用函数并合并数据
merged_df = merge_data_files(model_file, vna_data_file, evna_data_file, year_prefix)

# 保存合并后的数据
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_seasonal_data_fusion_DFT_IA.csv'
merged_df.to_csv(output_file, index=False)
print(f"数据已成功合并并保存到 {output_file}")
