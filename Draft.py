import pandas as pd

# 读取文件
file_path = '/DeepLearning/mnt/shixiansheng/data_fusion/output/Test.csv'
df = pd.read_csv(file_path)

# 指定 Period 列的值，这里以 '98th' 为例，你可以根据需要修改
specified_period = 'DJF_2011'

# 筛选出 Period 列等于指定值的行
df = df[df['Period'] == specified_period]

# 提取需要的列
columns_to_extract = ['ROW', 'COL', 'avna_ozone', 'model', 'vna_ozone', 'evna_ozone']
result = df[columns_to_extract]

# 检查 model 有值而 evna_ozone 没有值的行
model_has_value = result['model'].notna()  # model 列有效值
evna_has_no_value = result['evna_ozone'].isna()  # evna_ozone 列无值

# 找出 model 有值而 evna_ozone 没有值的行
model_with_valid_but_evna_without = result[model_has_value & evna_has_no_value][['ROW', 'COL', 'model']]

# 输出结果
if not model_with_valid_but_evna_without.empty:
    print("以下行中，model 列有有效值而 evna_ozone 列没有有效值：")
    print(model_with_valid_but_evna_without)
else:
    print("没有行中 model 列有有效值而 evna_ozone 列没有有效值。")