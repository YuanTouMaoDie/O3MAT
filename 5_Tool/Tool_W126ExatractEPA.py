import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import os


def save_to_csv(df, output_path):
    """ 将 DataFrame 保存为 CSV 文件 """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")


def extract_w126_nc_to_dataframe(nc_file):
    """
    从指定的 NetCDF 文件中提取 W126_VNA 数据，并返回 DataFrame。

    参数:
    - nc_file: NetCDF 文件的路径。

    返回:
    - 包含 W126_VNA 数据的 DataFrame。
    """
    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        rows = f.dimensions['ROW'].size  # 网格行数
        cols = f.dimensions['COL'].size  # 网格列数
        tstep = len(f.dimensions['TSTEP'])  # 时间步长

        # 提取 W126_VNA 变量
        vna_data = f.variables['W126_CMAQ'][:]

    # 假设每年 1 个时间步长，第十年就是2011 - 2002 + 1 - 1 =9个时间步长
    target_step = 9  # 2018 年对应的时间步长索引（从 0 开始计数）

    # 创建 DataFrame 来存储提取的数据
    data = {
        'ROW': [],
        'COL': [],
        'model': [],  # 这里直接命名为 VNA
        'Period': []
    }

    for row in range(rows):
        for col in range(cols):
            data['ROW'].append(row + 1)  # 索引从 1 开始
            data['COL'].append(col + 1)
            data['model'].append(vna_data[target_step, 0, row, col].item())
            data['Period'].append('W126')  # Period 固定为 'W126'

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    return df


# 定义路径
w126_nc_file = '/backupdata/data_EPA/EQUATES/W126/W126_CMAQ_March_October_VNA_2002_2019.nc'
output_file = '/DeepLearning/mnt/shixiansheng/data_fusion/output/W126/W126_Model_2011_March_October.csv'

# 处理 W126_VNA 数据并保存
w126_df = extract_w126_nc_to_dataframe(w126_nc_file)
save_to_csv(w126_df, output_file)