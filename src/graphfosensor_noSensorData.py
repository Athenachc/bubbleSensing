import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 1. INITIALIZE DATA STRUCTURES
number, magnitude, normal_force_sensor_value = [], [], []
vectors_x, vectors_y, vector_mean_x, vector_mean_y = [], [], [], []
new_magnitude, result_new_magnitude_plus = [], []

# 2. DATA LOADING AND PARSING
FILE_PATH = 'Sensor/Trial_20260713_221233/sensor.txt'
with open(FILE_PATH, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        # Handle cases where sensor data might be missing or empty
        number.append(int(parts[0]))
        # Use None if sensor column is empty/invalid
        nf_val = float(parts[1]) if parts[1] not in ['None', ''] else None
        normal_force_sensor_value.append(nf_val)
        
        magnitude.append(float(parts[2]))
        vectors_x.append(float(parts[3]))
        vectors_y.append(float(parts[4]))

        # Calculate running means
        vector_mean_x.append(np.mean(vectors_x))
        vector_mean_y.append(np.mean(vectors_y))

# Calculate Moving Average for Normal Force
new_magnitude_plus = [i * 0.009480253108 for i in magnitude]
n = 50
result_new_magnitude_plus = np.convolve(new_magnitude_plus, np.ones(n)/n, mode='same')

# Calculate Error (only where sensor data exists)
error_N = [
    (result_new_magnitude_plus[i] - val) if val is not None else None 
    for i, val in enumerate(normal_force_sensor_value)
]

# 3. PLOTTING
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Normal Force
axs[0].plot(number, result_new_magnitude_plus, label='Calculated', linewidth=2)
if any(v is not None for v in normal_force_sensor_value):
    axs[0].plot(number, normal_force_sensor_value, label='Sensor', linewidth=2, linestyle='--')
axs[0].set_title('Normal Force vs Frame')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Error
if any(v is not None for v in error_N):
    axs[1].plot(number, error_N, color='g', label='Error')
    axs[1].set_title('Normal Force Error')
else:
    axs[1].text(0.5, 0.5, 'No Sensor Data Available', ha='center', transform=axs[1].transAxes)
axs[1].grid(True)

# Plot 3: Shear Force
axs[2].plot(number, vector_mean_x, color='r', label='Calculated X')
axs[2].plot(number, vector_mean_y, color='b', label='Calculated Y')
axs[2].set_title('Shear Force vs Frame')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()