import os

# 定义变量映射
mapping = {
    "avna_ozone": "aVNA",
    "evna_ozone": "eVNA",
    "evna": "eVNA",
    "ds_ozone": "Downscaler",
    "model": "EQUATES",
    "harvard_ml": "Harvard ML"
}

# 定义基础路径
base_path = "/DeepLearning/mnt/shixiansheng/data_fusion/output/2011_CompareScatter"

# 遍历基础路径下的所有文件
for filename in os.listdir(base_path):
    if filename.endswith("_density.png"):
        # 拆分文件名
        parts = filename.split("_")
        year = parts[2]
        period = parts[4]

        # 查找变量名部分
        var_start_index = next((i for i, part in enumerate(parts) if "vs" in part), None)
        var_parts = parts[var_start_index:]

        # 替换变量名
        new_var_parts = []
        for part in var_parts:
            for old_var, new_var in mapping.items():
                part = part.replace(old_var, new_var)
            new_var_parts.append(part)

        # 去除多余部分
        new_var_str = "".join(new_var_parts).replace("_density.png", "")

        # 构建新的文件名
        new_filename = f"{year}_{period}_{new_var_str}.png"

        # 重命名文件
        old_file_path = os.path.join(base_path, filename)
        new_file_path = os.path.join(base_path, new_filename)
        try:
            os.rename(old_file_path, new_file_path)
            print(f"已将 {filename} 重命名为 {new_filename}")
        except FileNotFoundError:
            print(f"错误: 文件 {old_file_path} 未找到。")
        except Exception as e:
            print(f"错误: 重命名 {old_file_path} 时出现问题: {e}")
    