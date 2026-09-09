from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path("Sensor/poking/1_good")
sensor_files = list(base_dir.glob("**/sensor.txt"))

fixed_ratio_N = 0.01267076
n = 50

object_names = []
error_distributions = []

for file_path in sensor_files:
    magnitude = []
    normal_force_sensor_value = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ', 5)
            if len(parts) >= 3:
                try:
                    normal_force_sensor_value.append(float(parts[1].strip()))
                    magnitude.append(float(parts[2].strip()))
                except ValueError:
                    continue
                    
    if not magnitude or not normal_force_sensor_value:
        continue

    new_magnitude_plus = [i * fixed_ratio_N for i in magnitude]
    result_new_magnitude_plus = []
    for i in range(len(new_magnitude_plus)):
        window = new_magnitude_plus[i - n + 1 : i + 1]
        result_new_magnitude_plus.append(sum(window) / n)

    error_N = np.array(result_new_magnitude_plus) - np.array(normal_force_sensor_value)
    
    object_names.append(file_path.parent.name)
    error_distributions.append(error_N)

# Sort objects by mean absolute error
sorted_indices = sorted(range(len(error_distributions)), key=lambda k: np.mean(np.abs(error_distributions[k])))
object_names = [object_names[i] for i in sorted_indices]
error_distributions = [error_distributions[i] for i in sorted_indices]

plt.figure(figsize=(12, 6))
plt.boxplot(error_distributions, tick_labels=object_names, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red', linewidth=1.5))

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Error Distribution per Object (Ranked by Mean Absolute Error)', fontsize=11, fontweight='bold')
plt.ylabel('Error (Newton)', fontsize=10)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.grid(True, linestyle=':', alpha=0.7, axis='y')
plt.tight_layout()
plt.show()