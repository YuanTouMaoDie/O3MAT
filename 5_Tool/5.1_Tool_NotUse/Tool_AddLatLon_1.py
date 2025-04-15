import pandas as pd
import pandas as pd

# 读取第二个文件，过滤出 site_id, POCode, dateon, O3 列并且 O3 != -999 的行
file_path_2 = '/backupdata/data_EPA/aq_obs/routine/2004/AQS_hourly_data_2004.csv'
df_2 = pd.read_csv(file_path_2)
filtered_df = df_2[(df_2['O3'] != -999)][['site_id', 'POCode', 'dateon', 'O3']]
filtered_df['site_id'] = filtered_df['site_id'].astype(str)

# 统计输出文件中的唯一站点个数
output_unique_site_count = filtered_df['site_id'].nunique()

# 输出结果到指定文件
output_path = '/backupdata/data_EPA/aq_obs/routine/2004/AQS_hourly_data_2004_LatLon.csv'
filtered_df.to_csv(output_path, index=False)

print(f"输出文件中的唯一站点个数: {output_unique_site_count}")
print(f"结果已成功保存到 {output_path}")
    
