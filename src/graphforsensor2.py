import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 1. DATA LOADING
FILE_PATH = 'Sensor/Trial_20260713_213053/sensor.txt'
data = np.genfromtxt(FILE_PATH, delimiter=', ')

frame = data[:, 0]
normal_force_sensor_value = data[:, 1]
magnitude = data[:, 2]
vectors_x = data[:, 3]
vectors_y = data[:, 4]

# 2. CALCULATION LOGIC (Matching your original script)
# Running means for Shear Force
vector_mean_x = [np.mean(vectors_x[:i+1]) for i in range(len(vectors_x))]
vector_mean_y = [np.mean(vectors_y[:i+1]) for i in range(len(vectors_y))]

# Normal Force Scaling (Moving Average)
new_magnitude_plus = magnitude * 0.009480253108
n = 50
result_new_magnitude_plus = np.convolve(new_magnitude_plus, np.ones(n)/n, mode='same')

# Errors
error_N = result_new_magnitude_plus - normal_force_sensor_value

# 3. PLOTTING
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Normal Force
axs[0].plot(frame, result_new_magnitude_plus, label='Average', linewidth=2)
axs[0].plot(frame, normal_force_sensor_value, label='Sensor', linewidth=2)
axs[0].set_title('Normal Force vs Frame')
axs[0].legend()
axs[0].grid(True)
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(200))

# Plot 2: Normal Force Error
axs[1].plot(frame, error_N, color='g', linewidth=2)
axs[1].set_title('Normal Force Error vs Frame')
axs[1].grid(True)

# Plot 3: Shear Force
axs[2].plot(frame, vector_mean_x, color='r', label='Calculated', linewidth=2)
axs[2].plot(frame, vector_mean_y, color='b', label='Sensor', linewidth=2)
axs[2].set_title('Shear Force vs Frame')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()