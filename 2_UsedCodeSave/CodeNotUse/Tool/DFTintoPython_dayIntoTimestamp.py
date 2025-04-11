import os
import pandas as pd
import re

# 变量名映射字典 (如果需要特定映射)
variable_mapping = {
    'O3_aVNA': 'avna_ozone',
    'O3_eVNA': 'evna_ozone',
    'O3_VNA': 'vna_ozone',
    'O3_mod': 'vna_mod',
    'O3_Model':'model',
    # 可以根据需要添加其他映射
}

def merge_data_files(input_file, year_prefix, column_pattern=r'^[A-Za-z0-9_]+_d\d{4}$', date_format='%Y-%m-%d'):
    """
    合并数据文件，提取变量并处理日期参数，并将数据从宽格式转换为长格式。
    
    参数：
    input_file : str
        输入的 CSV 文件路径。
    year_prefix : str
        使用的年份前缀（如 2011），用于添加到 Period 列中。
    column_pattern : str
        用于匹配列名的正则表达式，默认匹配以字母或数字开头并包含日期参数的列名。
    date_format : str
        转换日期的格式，默认为 '%Y-%m-%d' 格式。
    
    返回：
    merged_df : pandas.DataFrame
        合并后的 DataFrame，数据已转为长格式，并包含 Timestamp 列。
    """
    # 读取输入数据
    df = pd.read_csv(input_file)

    # 使用正则表达式匹配列名，提取具有日期参数的列
    time_columns = [col for col in df.columns if re.match(column_pattern, col)]  # e.g., O3_aVNA_d0101

    # 创建一个空的DataFrame来存储最终的合并结果
    merged_df = pd.DataFrame()

    for period in time_columns:
        # 提取变量名和日期参数部分
        variable, date_param = period.split('_d')  # 假设列名为 "O3_aVNA_d0101"
        
        # 使用映射字典获取标准化后的变量名，如果没有映射则使用原名
        period_name = variable_mapping.get(variable, variable)  # 如果没有映射，则使用原名称
        
        # 格式化Period名称并加上年份前缀
        period_name = f"{year_prefix}_{period_name}"

        # 获取时间段数据
        period_data = df[period].dropna().reset_index(drop=True)

        # 生成时间戳
        timestamp = f"{year_prefix}-{date_param[:2]}-{date_param[2:]}"  # 例如 0101 -> 2011-01-01

        # 将数据按行存储
        temp_df = pd.DataFrame({
            'Timestamp': [timestamp] * len(period_data),
            period_name: period_data
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

    # 删除空列（没有数据的列）
    merged_df = merged_df.dropna(axis=1, how='all')

    # 去掉列名前的年份部分 (例如 2011_)
    merged_df.columns = [col.replace(f"{year_prefix}_", "") for col in merged_df.columns]

    # 以ROW和COL为唯一标识，合并每个ROW-COL组合的多列数据
    merged_df = merged_df.groupby(['ROW', 'COL', 'Timestamp'], as_index=False).first()

    # 在合并后的数据中按照Timestamp、ROW和COL排序
    merged_df = merged_df.sort_values(by=['Timestamp', 'ROW', 'COL']).reset_index(drop=True)

    # 交换ROW和COL的数值
    merged_df['ROW'], merged_df['COL'] = merged_df['COL'], merged_df['ROW']

    # 重新按照ROW和COL排序
    merged_df = merged_df.sort_values(by=['Timestamp', 'ROW', 'COL']).reset_index(drop=True)

    # 将动态生成的列名放到前面
    cols = ['ROW', 'COL', 'Timestamp'] + [col for col in merged_df.columns if col not in ['ROW', 'COL', 'Timestamp']]
    merged_df = merged_df[cols]

    return merged_df

# 输入文件路径
data_fusion_dir = "/DeepLearning/mnt/shixiansheng/data_fusion/output/DFT_Test"
input_file = os.path.join(data_fusion_dir, "1.csv")  # 修改为实际的文件名

# 自定义年份前缀
year_prefix = "2011"  # 可以修改为其他年份，比如 "2012" 等

# 调用函数并合并数据
merged_df = merge_data_files(input_file, year_prefix)

# 保存合并后的数据
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/DFT_Test/1_Trans.csv'
merged_df.to_csv(output_file, index=False)
print(f"数据已成功合并并保存到 {output_file}")
