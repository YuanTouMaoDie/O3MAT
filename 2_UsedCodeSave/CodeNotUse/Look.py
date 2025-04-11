import netCDF4 as nc

# 打开 NetCDF 文件
file_path = '/backupdata/data_EPA/EQUATES/DS_data/CMAQv532_DSFusion_12US1_2011.nc'  # 替换为实际的文件路径
dataset = nc.Dataset(file_path)

# 提取 MDA_O3 变量
mda_o3 = dataset.variables['MDA_O3']

# 获取特定位置的数据（TSTEP=182, LAY=0, ROW=1, COL=1）
value = mda_o3[182, 0, 149, 149]  # 根据索引提取数据
print(value)
