import pandas as pd

# Load the CSV file
file_path = '/backupdata/data_EPA/EQUATES/EQUATES_data/ds.input.aqs.o3.2011.csv'
data = pd.read_csv(file_path)

# Create a new DataFrame with the added row for each Site
new_data = data.copy()

# Add a new row for each unique site with Conc = 1 on the date 2011-01-01
site_counts = data['Site'].value_counts()
new_rows = pd.DataFrame({
    'Site': site_counts.index,
    'POC': 1,
    'Date': ['2011/1/1'] * len(site_counts),
    'Lat': [None] * len(site_counts),
    'Lon': [None] * len(site_counts),
    'Conc': [1] * len(site_counts)
})

# Use the first 'Lat' and 'Lon' values for each Site
first_lat_lon = data.groupby('Site').first()[['Lat', 'Lon']]

# Merge the new rows with the corresponding 'Lat' and 'Lon' values
new_rows['Lat'] = new_rows['Site'].map(first_lat_lon['Lat'])
new_rows['Lon'] = new_rows['Site'].map(first_lat_lon['Lon'])

# Append the new rows to the original data
data_with_new_rows = pd.concat([new_data, new_rows], ignore_index=True)

# Save the updated data to a CSV file
output_file_with_lat_lon = '/DeepLearning/mnt/shixiansheng/data_fusion/output/ds.input.aqs.o3.2011_FullDJF.csv'
data_with_new_rows.to_csv(output_file_with_lat_lon, index=False)
