import pandas as pd
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

def save_to_csv(df, output_path):
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存为 CSV 文件：{output_path}")

def extract_nc_to_dataframe(nc_file):
    # 手动指定年份
    year = '2011'

    # 打开 NetCDF 文件
    with Dataset(nc_file, 'r') as f:
        # 获取 ROW 和 COL 维度的大小
        rows = f.dimensions['ROW'].size
        cols = f.dimensions['COL'].size
        tstep = len(f.dimensions['TSTEP'])

        # 获取 MDA_O3_ERR 变量数据
        mda_o31 = f.variables['MDA8_O3'][:]

        # 构建一个 DataFrame 并提取全年的数据
        data = {
            'ROW': [],
            'COL': [],
            'model': [],
            'Timestamp': []
        }

        for i in tqdm(range(tstep)):
            # 生成日期
            date = pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            for row in range(rows):
                for col in range(cols):
                    data['ROW'].append(row + 1)  # 行和列从 1 开始
                    data['COL'].append(col + 1)
                    data['model'].append(mda_o31[i, 0, row, col].item())
                    data['Timestamp'].append(date_str)

        # 将数据保存到 DataFrame 中
        df = pd.DataFrame(data)

        # 输出 CSV 路径
        output_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Harvard_model_2011_daily.csv'
        save_to_csv(df, output_path)

# 输入.nc 文件路径
nc_file = '/backupdata/data_EPA/Harvard/unzipped_tifs/Harvard_O3MDA8_Regridded_grid_center_2011_12km.nc'

# 提取数据并保存
extract_nc_to_dataframe(nc_file)