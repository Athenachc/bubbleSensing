import matplotlib.pyplot as plt
import re
import numpy as np

shear_coefficient = 0.41

# Initialize lists to store data parsed from the sensor file
number = []
magnitude = []
normal_force_sensor_value = []
center_distance = []
vectors_x = []
vectors_y = []
vector_mean_x = []
vector_mean_x_inv = []
compare_org = []
vector_mean_y = []
new_magnitude = []
radio_N = []
radio_S = []
radio_S2 = []
error_N = []
error_S = []

# 1. DATA LOADING AND PARSING
# Read the file line by line, skipping the header or parsing the CSV format
# with open('Sensor/sensor.txt', 'r') as file:
with open('Sensor_set/Sensor_20260624_1/sensor.txt', 'r') as file:

    for line in file:
        # Create 6 empty lists for the 6 columns
        column_vectors = [[] for _ in range(6)]
        # Split string by comma to separate columns (Frame, NF, Mag, VecX, VecY)
        parts = line.strip().split(', ', 5)
        # Append the values to the corresponding lists
        number.append(int(parts[0].strip()))
        normal_force_sensor_value.append(float(parts[1].strip()))
        magnitude.append(float(parts[2].strip()))
        vectors_x.append(float(parts[3].strip()))
        vectors_y.append(float(parts[4].strip()))

        # Calculate means
        vector_mean_x.append(float(np.mean(vectors_x)))
        vector_mean_x_inv.append(-float(np.mean(vectors_x)))

        # Calculate y-coordinate mean
        vector_mean_y.append(float(np.mean(vectors_y)) if vectors_y else 0.0)

        # Calculate new magnitude (subtract mean from all vectors)
        mean_x = float(np.mean(vectors_x))
        mean_y = float(np.mean(vectors_y)) if vectors_y else 0.0

        # Create a coordinate array of all vectors read so far
        # Subtract the mean to normalize the displacement (zero-center the data)
        new_vector = np.array([vectors_x, vectors_y]).T
        new_vector[:, 0] -= mean_x
        new_vector[:, 1] -= mean_y
        
        # Sum the Euclidean distances (magnitudes) of the normalized vectors
        tmp = [np.sqrt(np.square(vec[0]) + np.square(vec[1])) for vec in new_vector]
        new_magnitude.append(float(np.sum(tmp)))

# Calculate ratios
# Calculate calibration ratios between the raw magnitude and sensor output
nf = float(np.sum(normal_force_sensor_value))
nm = float(np.sum(magnitude))
if nf == 0 or nm == 0:
    radio_N.append(0)
else:
    radio_N.append(nf / nm)

#sf = float(np.sum(shear_force_sensor_value))
#sm = float(np.sum(vector_mean_x))
#sm2 = float(np.sum(compare_org))

#if sf == 0 or sm == 0:
#    radio_S.append(0)
#else:
#    radio_S.append(sf / sm)

# if sf == 0 or sm2 == 0:
#     radio_S2.append(0)
# else:
#     radio_S2.append(sf / sm2)

# Scale values for comparison
new_magnitude_plus = [i * 0.00969 for i in magnitude]
vector_mean_x_plus = [i for i in vector_mean_x]                         # Compute error between calculated force and actual sensor feedback
#compare_org_plus = [i * shear_coefficient for i in compare_org]

# Calculate errors (Compute error between calculated force and actual sensor feedback)
error_N = [new_magnitude_plus[i] - normal_force_sensor_value[i] for i in range(len(normal_force_sensor_value))]
#error_S = [vector_mean_x_plus[i] - shear_force_sensor_value[i] for i in range(len(shear_force_sensor_value))]

print('\n')
print("radio_N: ", np.mean(radio_N))
print('\n')
# print("radio_S: ", radio_S)
# print('\n')
# print("radio_S2: ", radio_S2)
# print('\n')

# Create plots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Normal Force comparison
axs[0].plot(number, new_magnitude_plus, marker='o', linestyle='-', color='r', linewidth=2, markersize=2, label='Calculated')
axs[0].plot(number, normal_force_sensor_value, marker='o', linestyle='-', color='b', linewidth=2, markersize=2, label='Sensor')
axs[0].set_title('Normal Force vs Frame')
axs[0].set_xlabel('Frame (ticks)')
axs[0].set_ylabel('Normal Force (Newton)')
axs[0].legend()
axs[0].grid()

# Plot 2: Normal Force Error
axs[1].plot(number, error_N, marker='o', linestyle='-', color='g', linewidth=2, markersize=2)
axs[1].set_title('Normal Force Error vs Frame')
axs[1].set_xlabel('Frame (ticks)')
axs[1].set_ylabel('Error (N)')
axs[1].grid()

# # Plot 3: Shear Force comparison
# axs[1].plot(number, vector_mean_x_plus, marker='o', linestyle='-', color='r', linewidth=2, markersize=2, label='Calculated')
# #axs[1].plot(number, shear_force_sensor_value, marker='o', linestyle='-', color='b', linewidth=2, markersize=2, label='Sensor')
# axs[1].set_title('Shear Force vs Frame')
# axs[1].set_xlabel('Frame (ticks)')
# axs[1].set_ylabel('Shear Force (Newton)')
# axs[1].legend()
# axs[1].grid()

# # # Plot 4: Shear Force Error
# # axs[3].plot(number, error_S, marker='o', linestyle='-', color='y', linewidth=2, markersize=2)
# # axs[3].set_title('Shear Force Error vs Frame')
# # axs[3].set_xlabel('Frame (ticks)')
# # axs[3].set_ylabel('Error (N)')
# # axs[3].grid()

# # Plot 5: Original Magnitude
# axs[2].plot(number, magnitude, marker='o', linestyle='-', color='g', linewidth=3, markersize=3)
# axs[2].set_title('Raw Magnitude vs Frame')
# axs[2].set_xlabel('Frame (ticks)')
# axs[2].set_ylabel('Magnitude (Pixels)')
# axs[2].grid()

fig.tight_layout()
plt.show()
