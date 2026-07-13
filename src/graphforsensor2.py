import matplotlib.pyplot as plt
import numpy as np

# 1. DATA LOADING
FILE_PATH = 'Sensor/Trial_20260713_213053/sensor.txt'
# Using np.genfromtxt to handle your comma-separated sensor data
data = np.genfromtxt(FILE_PATH, delimiter=', ')

frame = data[:, 0]
normal_force_sensor_value = data[:, 1]
magnitude = data[:, 2]
vectors_x = data[:, 3]
vectors_y = data[:, 4]

# 2. CALCULATION LOGIC
vector_mean_x = [np.mean(vectors_x[:i+1]) for i in range(len(vectors_x))]
vector_mean_y = [np.mean(vectors_y[:i+1]) for i in range(len(vectors_y))]

new_magnitude_plus = magnitude * 0.009480253108
n = 50
result_new_magnitude_plus = np.convolve(new_magnitude_plus, np.ones(n)/n, mode='same')
error_N = result_new_magnitude_plus - normal_force_sensor_value

# 3. PLOTTING FUNCTION
def plot_all_marker_displacements(frame_idx, data_array):
    # 1. Extract raw data
    u_raw = data_array[frame_idx, 5:14]
    v_raw = data_array[frame_idx, 14:23]
    
    # 2. Safety Padding: If we have < 9 points, pad with zeros
    def ensure_3x3(arr):
        if arr.size < 9:
            new_arr = np.zeros(9)
            new_arr[:arr.size] = arr
            return new_arr.reshape(3, 3)
        return arr.reshape(3, 3)
    
    u = ensure_3x3(u_raw)
    v = ensure_3x3(v_raw)
    
    # 3. Create the 3x3 coordinate grid
    x = np.arange(3)
    y = np.arange(3)
    X, Y = np.meshgrid(x, y)
    
    fig_m, ax_m = plt.subplots(figsize=(6, 6))
    
    # 4. Plot
    ax_m.quiver(X, Y, u, v, angles='xy', scale_units='xy', scale=1, color='purple')
    
    ax_m.set_xlim(-0.5, 2.5); ax_m.set_ylim(-0.5, 2.5)
    ax_m.invert_yaxis()
    ax_m.set_title(f"Marker Displacements - Frame {int(data_array[frame_idx, 0])}")
    ax_m.grid(True)
    return fig_m

# 4. MAIN PLOTTING
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

axs[0].plot(frame, result_new_magnitude_plus, label='Average', linewidth=2)
axs[0].plot(frame, normal_force_sensor_value, label='Sensor', linewidth=2)
axs[0].set_title('Normal Force vs Frame')
axs[0].legend(); axs[0].grid(True)

axs[1].plot(frame, error_N, color='g', linewidth=2)
axs[1].set_title('Normal Force Error vs Frame'); axs[1].grid(True)

axs[2].plot(frame, vector_mean_x, color='r', label='Calculated', linewidth=2)
axs[2].plot(frame, vector_mean_y, color='b', label='Sensor', linewidth=2)
axs[2].set_title('Shear Force vs Frame')
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()

# Call the marker plot for a specific frame
plot_all_marker_displacements(706, data)

plt.show()