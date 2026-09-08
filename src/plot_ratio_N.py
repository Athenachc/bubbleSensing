from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Define the directory containing your trial folders
base_dir = Path("Sensor/5mm_2.5ml_v3/")  # Adjust this path as needed
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} sensor files to process.")

trial_data = []
all_ratios = []

# 1. Parse all trials and compute individual ratios
for file_path in sensor_files:
    numbers = []
    nf_values = []
    magnitudes = []
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ", 5)
            if len(parts) >= 5:
                try:
                    numbers.append(int(parts[0].strip()))
                    nf_values.append(float(parts[1].strip()))
                    magnitudes.append(float(parts[2].strip()))
                except ValueError:
                    continue
                    
    if not nf_values or not magnitudes:
        continue
        
    nf_sum = np.sum(nf_values)
    mag_sum = np.sum(magnitudes)
    
    if mag_sum > 0:
        ratio = nf_sum / mag_sum
        all_ratios.append(ratio)
        trial_data.append({
            "path": file_path,
            "number": numbers,
            "normal_force": nf_values,
            "magnitude": magnitudes
        })

# 2. Compute the best global ratio_N as the mean across all trials
best_radio_N = np.mean(all_ratios)
print(f"Best global radio_N (mean of {len(all_ratios)} trials): {best_radio_N:.8f}")

# 3. Plot all trials using the unified best ratio_N and show errors
fig, axs = plt.subplots(2, 1, figsize=(10, 9))

n_window = 50  # Window size for moving average

for data in trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    numbers = data["number"]
    
    # Calculate force using the unified best ratio
    calculated_nf = mag * best_radio_N
    
    # Apply moving average filter
    if len(calculated_nf) >= n_window:
        result_new_magnitude_plus = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        plot_numbers = numbers[n_window-1:]
        plot_sensor = nf_sensor[n_window-1:]
    else:
        result_new_magnitude_plus = calculated_nf
        plot_numbers = numbers
        plot_sensor = nf_sensor
        
    # Calculate error between calculated force and actual sensor feedback
    error_N = result_new_magnitude_plus - plot_sensor
    
    trial_name = data["path"].parent.name
    
    # Plot Normal Force comparison
    axs[0].plot(plot_numbers, result_new_magnitude_plus, linestyle='-', alpha=0.6, label=f'{trial_name}')
    
    # Plot Normal Force Error
    axs[1].plot(plot_numbers, error_N, linestyle='-', alpha=0.6, label=f'{trial_name}')

# Plot actual sensor baseline reference on top if needed, or leave individual curves clean
axs[0].set_title(f'Normal Force Comparison Across Trials (Unified radio_N = {best_radio_N:.6f})')
axs[0].set_xlabel('Frame (ticks)')
axs[0].set_ylabel('Normal Force (Newton)')
axs[0].grid(True)
if len(trial_data) <= 6:
    axs[0].legend(loc='upper left', fontsize='small', framealpha=0.7)

axs[1].set_title('Normal Force Error vs Frame (Calculated - Sensor)')
axs[1].set_xlabel('Frame (ticks)')
axs[1].set_ylabel('Error (N)')
axs[1].axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Error')
axs[1].grid(True)
if len(trial_data) <= 6:
    axs[1].legend(loc='upper left', fontsize='small', framealpha=0.7)

fig.tight_layout()
plt.show()