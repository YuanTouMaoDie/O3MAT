import csv
import multiprocessing
from functools import partial
from tqdm import tqdm

# 定义 Line 类，用于存储时区边界线的相关信息
class Line:
    def __init__(self, npts, name, offset, xmin, xmax, ymin, ymax, x, y):
        # 边界线上点的数量
        self.npts = npts
        # 时区的名称
        self.name = name
        # 时区的偏移量
        self.offset = offset
        # 边界线的最小经度
        self.xmin = xmin
        # 边界线的最大经度
        self.xmax = xmax
        # 边界线的最小纬度
        self.ymin = ymin
        # 边界线的最大纬度
        self.ymax = ymax
        # 边界线上所有点的经度列表
        self.x = x
        # 边界线上所有点的纬度列表
        self.y = y


# 计算给定经度 xx 与边界线交点的纬度值
def get_values(xx, npts, x, y):
    # 用于存储交点的纬度值
    yy = []
    # 遍历边界线上相邻的点对
    for i in range(npts - 1):
        # 判断给定经度是否在相邻两点的经度范围内
        if (xx >= x[i] and xx < x[i + 1]) or (xx <= x[i] and xx > x[i + 1]):
            # 计算斜率，如果两点经度相同则斜率为 1.0
            slope = 1.0 if x[i] == x[i + 1] else (y[i] - y[i + 1]) / (x[i] - x[i + 1])
            # 通过线性插值计算交点的纬度值并添加到列表中
            yy.append(y[i] + slope * (xx - x[i]))
    return yy


# 判断给定纬度 x 是否在一组纬度值 values 所定义的区域内
def in_area(x, values):
    # 对纬度值进行排序
    values = sorted(values)
    # 以步长 2 遍历排序后的纬度值，检查 x 是否在相邻两个值之间
    for i in range(0, len(values) - 1, 2):
        if x >= values[i] and x <= values[i + 1]:
            return True
    return False


# 根据给定的经纬度计算时区偏移量
def get_tz(longitude, latitude, tz_file='output/Region/tz.csv'):
    global first_time, lines
    # 如果是第一次调用该函数，需要读取时区数据文件
    if first_time:
        first_time = False
        # 用于存储所有时区边界线的信息
        lines = []
        try:
            # 打开时区数据文件
            with open(tz_file, 'r') as file:
                # 创建 CSV 读取器
                reader = csv.reader(file)
                while True:
                    try:
                        # 读取一行数据
                        line = next(reader)
                    except StopIteration:
                        # 读取完文件则跳出循环
                        break
                    # 解析边界线上点的数量
                    npts = int(line[0])
                    # 解析时区偏移量
                    offset = float(line[1])
                    # 解析时区名称
                    name = line[2]
                    # 用于存储边界线上点的经度
                    lon = []
                    # 用于存储边界线上点的纬度
                    lat = []
                    # 读取边界线上每个点的经纬度
                    for _ in range(npts):
                        point_line = next(reader)
                        lon.append(float(point_line[0]))
                        lat.append(float(point_line[1]))
                    # 计算边界线的最小经度
                    xmin = min(lon)
                    # 计算边界线的最大经度
                    xmax = max(lon)
                    # 计算边界线的最小纬度
                    ymin = min(lat)
                    # 计算边界线的最大纬度
                    ymax = max(lat)
                    # 创建 Line 对象并添加到 lines 列表中
                    line_obj = Line(npts, name, offset, xmin, xmax, ymin, ymax, lon, lat)
                    lines.append(line_obj)
        except FileNotFoundError:
            # 若文件未找到，打印错误信息并返回 None
            print(f"**ERROR** Cannot open time zone data file: {tz_file}")
            return None

    # 用于记录交点的数量
    nx = 0
    # 用于存储交点的纬度值
    xsec = []
    # 用于存储交点对应的时区偏移量
    ysec = []
    # 遍历所有时区边界线
    for line in lines:
        # 判断给定经度是否在边界线的经度范围内
        if line.xmin <= longitude <= line.xmax:
            # 计算给定经度与边界线交点的纬度值
            lat_values = get_values(longitude, line.npts, line.x, line.y)
            # 判断交点数量是否不少于 2 且给定纬度在边界线的纬度范围内
            if len(lat_values) >= 2 and line.ymin <= latitude <= line.ymax:
                # 判断给定纬度是否在交点纬度所定义的区域内
                if in_area(latitude, lat_values):
                    # 若满足条件，返回对应的时区偏移量（取反）
                    print(f"Found match in zone: {line.name}, offset: {line.offset}")
                    return -int(line.offset)
            # 将交点的纬度值和对应的时区偏移量添加到列表中
            for val in lat_values:
                nx += 1
                xsec.append(val)
                ysec.append(line.offset)

    # 如果有多个交点
    if nx > 1:
        # 对交点的纬度值和对应的时区偏移量进行排序
        sorted_pairs = sorted(zip(xsec, ysec))
        xsec = [x for x, _ in sorted_pairs]
        ysec = [y for _, y in sorted_pairs]
        # 遍历排序后的交点纬度值
        for i in range(len(xsec) - 1):
            # 判断给定纬度是否在相邻两个交点纬度之间
            if latitude >= xsec[i] and latitude <= xsec[i + 1]:
                # 判断相邻两个交点对应的时区偏移量是否相同且距离小于 2.0 度
                if ysec[i] == ysec[i + 1] and xsec[i + 1] - xsec[i] < 2.0:
                    # 若满足条件，返回对应的时区偏移量（取反）
                    print(f"Found match between zones, offset: {ysec[i]}")
                    return -int(ysec[i])

    # 若以上条件都不满足，根据经度计算默认的时区偏移量
    long = abs(longitude)
    tz_offset = int((long + 7.5) / 15)
    # 如果经度大于 0，对时区偏移量取反
    if longitude > 0:
        tz_offset = -tz_offset
    print(f"Using default offset based on longitude: {tz_offset}")
    return tz_offset


def process_row(row, tz_file='output/Region/tz.csv'):
    lon = float(row[2])  # 经纬度的索引从 2 开始，因为第一列是 site_id
    lat = float(row[1])
    # 不加负号，就为正
    tz_offset = -get_tz(lon, lat, tz_file)
    gmt_offset = tz_offset
    row.append(str(gmt_offset))
    return row


def process_data(input_file, output_file):
    global first_time
    first_time = True
    global lines
    lines = []

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        header.append('python_gmt_offset')
        new_rows = [header]

        rows = []
        for row in reader:
            rows.append(row)

        pool = multiprocessing.Pool()
        process_row_partial = partial(process_row, tz_file='output/Region/tz.csv')
        # 使用 pool.map 保证顺序
        results = pool.map(process_row_partial, rows)
        new_rows.extend(results)
        pool.close()
        pool.join()

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(new_rows)

    # 统计 python_gmt_offset 取值的种类数量，以及拥有这些值的网格数量
    gmt_offset_count = {}
    for row in new_rows[1:]:  # 跳过标题行
        gmt_offset = row[-1]
        if gmt_offset:
            gmt_offset_count[gmt_offset] = gmt_offset_count.get(gmt_offset, 0)

    num_gmt_offsets = len(gmt_offset_count)
    print(f"python_gmt_offset 取值的种类数量: {num_gmt_offsets}")
    for offset, count in gmt_offset_count.items():
        print(f"python_gmt_offset 值为 {offset} 的网格数量: {count}")


input_file = 'output/Region/MonitorsTimeRegion_Filter_ST.csv'
output_file = 'output/Region/MonitorsTimeRegion_Filter_ST_QA.csv'
process_data(input_file, output_file)