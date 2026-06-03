import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Sensor/sensor.txt', header=None, names=['number', 'pos', 'sensor_value', 'magnitude', 'vector_x', 'vector_y', 'center_distance'])

# Extract columns
number = data['number']
pos = data['pos']
sensor_value = data['sensor_value']
magnitude = data['magnitude']
vector_x = data['vector_x']
vector_y = data['vector_y']
center_distance = data['center_distance']

# Create figure and axes
fig, axs = plt.subplots(5, 1, figsize=(10, 15))

# Plot for sensor value
axs[0].plot(number, sensor_value, marker='o', linestyle='-', color='b')
axs[0].set_title('Sensor Value vs Number')
axs[0].set_xlabel('Number')
axs[0].set_ylabel('Sensor Value')
axs[0].grid()

# Plot for Magnitude
axs[1].plot(number, magnitude, marker='o', linestyle='-', color='g')
axs[1].set_title('Magnitude vs Number')
axs[1].set_xlabel('Number')
axs[1].set_ylabel('Magnitude')
axs[1].grid()

axs[2].plot(number, vector_x, marker='o', linestyle='-', color='g')
axs[2].set_title('Vector_X vs Number')
axs[2].set_xlabel('Number')
axs[2].set_ylabel('Vector_X')
axs[2].grid()

axs[3].plot(number, vector_y, marker='o', linestyle='-', color='g')
axs[3].set_title('Vector_Y vs Number')
axs[3].set_xlabel('Number')
axs[3].set_ylabel('Vector_Y')
axs[3].grid()

axs[4].plot(number, pos, marker='o', linestyle='', color='g')
axs[4].set_title('Pos vs Number')
axs[4].set_xlabel('Number')
axs[4].set_ylabel('Pos')
axs[4].grid()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

