from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the base directory containing all object folders
base_dir = Path("Sensor/poking/1_good")
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} object trial files to evaluate.\n")

if not sensor_files:
    print(f"No sensor files found in {base_dir}!")
    exit()

fixed_ratio_N = 0.01267076  # for 2.5ml 5mm camera bubble
n = 50                      # Window size for moving average

trial_results = []

# Loop through every sensor.txt file found in the directory
for file_path in sensor_files:
    number = []
    magnitude = []
    normal_force_sensor_value = []
    vectors_x = []
    vectors_y = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ', 5)
            if len(parts) >= 5:
                try:
                    number.append(int(parts[0].strip()))
                    normal_force_sensor_value.append(float(parts[1].strip()))
                    magnitude.append(float(parts[2].strip()))
                    vectors_x.append(float(parts[3].strip()))
                    vectors_y.append(float(parts[4].strip()))
                except ValueError:
                    continue
                    
    if not normal_force_sensor_value or not magnitude:
        continue

    # Scale values using your fixed calibration ratio
    new_magnitude_plus = [i * fixed_ratio_N for i in magnitude]

    # Exact moving average logic from your script
    result_new_magnitude_plus = []
    for i in range(len(new_magnitude_plus)):
        window = new_magnitude_plus[i - n + 1 : i + 1]
        result_new_magnitude_plus.append(sum(window) / n)

    # Calculate error
    error_N = [result_new_magnitude_plus[i] - normal_force_sensor_value[i] for i in range(len(normal_force_sensor_value))]
    
    # Calculate RMSE for sorting the grid
    sensor_arr = np.array(normal_force_sensor_value)
    detected_arr = np.array(result_new_magnitude_plus)
    rmse = np.sqrt(np.mean(np.square(detected_arr - sensor_arr)))

    trial_results.append({
        "object_name": file_path.parent.name,
        "number": number,
        "result_new_magnitude_plus": detected_arr,
        "normal_force_sensor_value": sensor_arr,
        "rmse": rmse
    })

# Sort trials by RMSE (lowest to highest)
trial_results.sort(key=lambda x: x["rmse"])

# Plot all objects in a 3x4 grid using your exact loop logic
fig, axs = plt.subplots(3, 4, figsize=(18, 11))
axs = axs.flatten()

for idx, trial in enumerate(trial_results):
    ax = axs[idx]
    ax.plot(trial["number"], trial["result_new_magnitude_plus"], linestyle='-', linewidth=1.5, label='Average (Detected)', color='dodgerblue')
    ax.plot(trial["number"], trial["normal_force_sensor_value"], linestyle='-', linewidth=1.5, label='Sensor (Ground Truth)', color='darkorange')
    ax.set_title(f"{trial['object_name']}\nRMSE: {trial['rmse']:.4f} N", fontsize=9, fontweight='bold')
    ax.set_ylabel('Force (N)', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', labelsize=8)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.7, edgecolor='none')

# Hide any empty subplots if fewer than 12
for j in range(len(trial_results), len(axs)):
    fig.delaxes(axs[j])

fig.suptitle("Normal Force Comparison Across All Objects (Exact User Logic)", fontsize=13, fontweight='bold')
fig.tight_layout()
plt.show()