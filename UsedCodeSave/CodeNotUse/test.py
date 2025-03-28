

import rasterio
import matplotlib.pyplot as plt

# Open the GeoTIFF file
file_path = '/DeepLearning/mnt/Devin/data/harvard/Ensemble_predictions_O3_USGrid_20110101_20110101.tif'
dataset = rasterio.open(file_path)

# Read the first band of the GeoTIFF file
data = dataset.read(1)

# Display the data
plt.figure(figsize=(10, 10))
plt.imshow(data, cmap='viridis')  # You can choose different color maps
plt.colorbar(label='O3 Ensemble Prediction')  # Add a colorbar with a label
plt.title('Ensemble Predictions O3 US Grid on 2011-01-01')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Close the dataset
dataset.close()
