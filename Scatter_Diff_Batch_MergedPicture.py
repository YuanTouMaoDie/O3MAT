import os
from PIL import Image
import re

# -------------------- 工具函数 --------------------
def extract_key_period(period):
    """
    Extract key period (e.g., JFM, AMJ) from the full period string.
    """
    key_periods = ["DJF", "MAM", "JJA", "SON", 'Annual', 'Apr-Sep', 'top-10', 'W126']
    for key in key_periods:
        if key in period:
            return key
    return None


def get_axis_label(filename, variable):
    """
    根据文件名和变量生成轴标签。
    - 如果变量为 'model'，根据文件名判断是 'Harvard ML' 还是 'EQUATES'。
    - 其他情况根据变量名确定标签，如 'vna_ozone' 对应 'VNA'。
    """
    if 'model' in variable:
        return "EQUATES"
    elif "harvard_ml" in variable:
        return "Harvard ML"
    elif "evna_ozone" in variable:
        return "eVNA"
    elif "avna_ozone" in variable:
        return "aVNA"
    elif "ds_ozone" in variable:
        return "Downscaler"
    elif "vna_ozone" in variable:
        return "VNA"
    elif "Conc" in variable:
        return "Monitor"
    elif "O3" in variable:
        return "Monitor (Hourly)"
    return "unknown"


# -------------------- 图片合并函数 --------------------
def merge_images(image_paths, output_path, spacing=20):
    """
    横向合并多张图片为一张图片，并设置图片之间的间距，用白色填充。

    :param image_paths: 单张图片的路径列表
    :param output_path: 合并后图片的保存路径
    :param spacing: 图片之间的间距，单位为像素，默认为 20
    """
    # 检查所有输入图片是否存在
    missing_images = [path for path in image_paths if not os.path.exists(path)]
    if missing_images:
        print(f"错误: 以下图片不存在: {missing_images}")
        return False
    
    # 读取所有图片
    try:
        images = [Image.open(path) for path in image_paths]
    except Exception as e:
        print(f"错误: 打开图片时出错: {e}")
        return False
    
    # 确保所有图片尺寸一致（使用第一张图片的尺寸）
    width, height = images[0].size
    for img in images[1:]:
        if img.size != (width, height):
            print(f"警告: 图片尺寸不一致，将调整为 {width}x{height}")
            img = img.resize((width, height))
    
    # 计算合并后图片的总宽度，考虑间距
    total_width = width * len(images) + spacing * (len(images) - 1)
    
    # 创建合并图片，宽度为所有图片宽度与间距之和，高度为单张图片的高度
    merged_image = Image.new('RGB', (total_width, height), color='white')
    
    # 粘贴每张图片到合并图片的对应位置，并添加间距
    current_x = 0
    for i, image in enumerate(images):
        merged_image.paste(image, (current_x, 0))
        current_x += width + spacing
    
    # 保存合并图片
    try:
        merged_image.save(output_path)
        print(f"合并图片已保存到 {output_path}")
        return True
    except Exception as e:
        print(f"错误: 保存合并图片时出错: {e}")
        return False


# -------------------- 按周期和组合并图片 --------------------
def merge_images_by_period_and_group(all_period_images, output_dir, groups, spacing=20):
    """
    按周期和指定的变量组合并图片
    
    :param all_period_images: 所有图片信息，按周期分组
    :param output_dir: 输出目录
    :param groups: 定义的变量组
    :param spacing: 图片间距
    """
    # 为每个组创建一个文件夹
    for group_idx, group in enumerate(groups):
        group_dir = os.path.join(output_dir, f"Group_{group_idx+1}")
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
            print(f"创建组文件夹: {group_dir}")
        
        # 按周期处理
        for period, image_info_list in all_period_images.items():
            # 筛选属于当前组的图片
            group_images = []
            for info in image_info_list:
                path, year, (x_col, y_col) = info
                # 检查这对变量是否在当前组中
                if (x_col, y_col) in group or (y_col, x_col) in group:
                    group_images.append(info)
            
            # 如果找到了属于该组的图片，则合并它们
            if group_images:
                image_paths = [info[0] for info in group_images]
                # 生成组名
                group_name = "_vs_".join([f"{get_axis_label('', pair[0])}_{get_axis_label('', pair[1])}" for pair in group])
                output_path = os.path.join(group_dir, f"Merged_{period}_{group_name}.png")
                
                # 检查合并图片是否已存在
                if os.path.exists(output_path):
                    print(f"合并图片已存在，跳过生成: {output_path}")
                    continue
                
                print(f"正在合并 {period} 的组 {group_idx+1} 图片: {image_paths}")
                if not merge_images(image_paths, output_path, spacing=spacing):
                    print(f"合并失败: {period} 的组 {group_idx+1}")


# -------------------- 主函数示例 --------------------
if __name__ == "__main__":
    # 示例：如何使用图片合并功能
    
    # 定义已存在的图片路径（按周期分组）
    # 格式：{周期: [(图片路径, 年份, (x变量, y变量)), ...]}
    all_period_images = {
        "2023_DJF": [
            ("/path/to/2023_DJF_model_vs_vna.png", "2023", ("model", "vna_ozone")),
            ("/path/to/2023_DJF_model_vs_evna.png", "2023", ("model", "evna_ozone")),
            ("/path/to/2023_DJF_model_vs_avna.png", "2023", ("model", "avna_ozone")),
        ],
        "2023_JJA": [
            ("/path/to/2023_JJA_model_vs_vna.png", "2023", ("model", "vna_ozone")),
            ("/path/to/2023_JJA_model_vs_evna.png", "2023", ("model", "evna_ozone")),
            ("/path/to/2023_JJA_model_vs_avna.png", "2023", ("model", "avna_ozone")),
        ]
    }
    
    # 输出目录
    output_dir = "/path/to/merged_images"
    
    # 定义合并组（与原代码相同）
    groups = [
        [('model','vna_ozone'), ('model','evna_ozone'), ('model','avna_ozone'), ('model','ds_ozone'), ('model','harvard_ml')],
        [('vna_ozone', 'evna_ozone'), ('vna_ozone', 'avna_ozone'), ('vna_ozone', 'ds_ozone'), ('vna_ozone', 'harvard_ml'), ('ds_ozone', 'harvard_ml')],
        [('evna_ozone', 'avna_ozone'), ('evna_ozone', 'ds_ozone'), ('evna_ozone', 'harvard_ml'), ('avna_ozone', 'ds_ozone'), ('avna_ozone', 'harvard_ml')]
    ]
    
    # 执行合并
    merge_images_by_period_and_group(all_period_images, output_dir, groups, spacing=30)
    
    print("图片合并完成！")