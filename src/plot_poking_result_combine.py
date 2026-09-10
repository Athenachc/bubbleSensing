from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

object_names = []
error_distributions = []
trial_data = []

# Load and parse all trials once
for file_path in sensor_files:
    number = []
    magnitude = []
    normal_force_sensor_value = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ', 5)
            if len(parts) >= 3:
                try:
                    number.append(int(parts[0].strip()))
                    normal_force_sensor_value.append(float(parts[1].strip()))
                    magnitude.append(float(parts[2].strip()))
                except ValueError:
                    continue
                    
    if not magnitude or not normal_force_sensor_value:
        continue

    # Scale values using fixed calibration ratio
    new_magnitude_plus = [i * fixed_ratio_N for i in magnitude]

    # Moving average logic
    result_new_magnitude_plus = []
    for i in range(len(new_magnitude_plus)):
        window = new_magnitude_plus[i - n + 1 : i + 1]
        result_new_magnitude_plus.append(sum(window) / n)

    # Calculate error
    error_N = np.array(result_new_magnitude_plus) - np.array(normal_force_sensor_value)
    
    object_name = file_path.parent.name
    object_names.append(object_name)
    error_distributions.append(error_N)
    
    trial_data.append({
        "object_name": object_name,
        "number": number,
        "error_N": error_N
    })

# Sort objects by mean absolute error for the box plot ranking
sorted_indices = sorted(range(len(error_distributions)), key=lambda k: np.mean(np.abs(error_distributions[k])))
sorted_object_names = [object_names[i] for i in sorted_indices]
sorted_error_distributions = [error_distributions[i] for i in sorted_indices]

# ==========================================
# WINDOW 1: Box Plot of Error Distribution
# ==========================================
plt.figure(1, figsize=(12, 6))
plt.boxplot(sorted_error_distributions, tick_labels=sorted_object_names, patch_artist=True, 
            showfliers=False,
            boxprops=dict(facecolor='lightblue', color='black', linewidth=1.2),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2))

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Error Distribution per Object (Ranked by Mean Absolute Error - Outliers Hidden)', fontsize=11, fontweight='bold')
plt.ylabel('Error (Newton)', fontsize=10)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.grid(True, linestyle=':', alpha=0.7, axis='y')
plt.tight_layout()

# ==========================================
# WINDOW 2: Line Graph of Error vs Frame
# ==========================================
plt.figure(2, figsize=(12, 7))
for trial in trial_data:
    plt.plot(trial["number"], trial["error_N"], linestyle='-', linewidth=1.2, alpha=0.8, label=trial["object_name"])

plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Normal Force Error vs Frame (All 12 Objects)', fontsize=12, fontweight='bold')
plt.xlabel('Frame (ticks)', fontsize=10)
plt.ylabel('Error (Newton)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.7)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
plt.tight_layout()

# Display both separate windows simultaneously
plt.show()