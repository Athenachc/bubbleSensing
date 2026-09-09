from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing your trial folders
base_dir = Path("Sensor/5mm_2.5ml_v3")
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} total sensor files to evaluate.")

trial_data = []
all_ratios = []

# 1. Parse all trials
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

# Initial rough global ratio to test errors
initial_radio_N = np.mean(all_ratios)

# 2. Filter out trials where the maximum absolute error exceeds 0.4 N
error_threshold = 0.345
n_window = 50
valid_trial_data = []
filtered_ratios = []

for data in trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    numbers = data["number"]
    
    calculated_nf = mag * initial_radio_N
    
    if len(calculated_nf) >= n_window:
        result_new_magnitude_plus = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        plot_sensor = np.array(nf_sensor[n_window-1:])
    else:
        result_new_magnitude_plus = calculated_nf
        plot_sensor = np.array(nf_sensor)
        
    error_N = result_new_magnitude_plus - plot_sensor
    max_error = np.max(np.abs(error_N))
    
    trial_name = data["path"].parent.name
    if max_error <= error_threshold:
        valid_trial_data.append(data)
        filtered_ratios.append(np.sum(nf_sensor) / np.sum(mag))
    else:
        print(f"Excluded [{trial_name}] -> Max Error: {max_error:.3f} N (exceeds {error_threshold} N threshold)")

# 3. Compute the refined best ratio_N using only the filtered/valid trials
if filtered_ratios:
    best_radio_N = np.mean(filtered_ratios)
else:
    best_radio_N = initial_radio_N

print(f"\nSuccessfully kept {len(valid_trial_data)} trials.")
print(f"Refined best global radio_N: {best_radio_N:.8f}")

# 4. Plot only the valid trials
fig, axs = plt.subplots(2, 1, figsize=(10, 9))

for data in valid_trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    numbers = data["number"]
    
    calculated_nf = mag * best_radio_N
    
    if len(calculated_nf) >= n_window:
        result_new_magnitude_plus = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        plot_numbers = numbers[n_window-1:]
        plot_sensor = nf_sensor[n_window-1:]
    else:
        result_new_magnitude_plus = calculated_nf
        plot_numbers = numbers
        plot_sensor = nf_sensor
        
    error_N = result_new_magnitude_plus - plot_sensor
    trial_name = data["path"].parent.name
    
    axs[0].plot(plot_numbers, result_new_magnitude_plus, linestyle='-', alpha=0.6, label=f'{trial_name}')
    axs[1].plot(plot_numbers, error_N, linestyle='-', alpha=0.6, label=f'{trial_name}')

axs[0].set_title(f'Normal Force Comparison (Filtered Trials, Unified radio_N = {best_radio_N:.6f})')
axs[0].set_xlabel('Frame (ticks)')
axs[0].set_ylabel('Normal Force (Newton)')
axs[0].grid(True)
if len(valid_trial_data) <= 6:
    axs[0].legend(loc='upper left', fontsize='small', framealpha=0.7)

axs[1].set_title('Normal Force Error vs Frame (Calculated - Sensor)')
axs[1].set_xlabel('Frame (ticks)')
axs[1].set_ylabel('Error (N)')
axs[1].axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Error')
axs[1].grid(True)
if len(valid_trial_data) <= 6:
    axs[1].legend(loc='upper left', fontsize='small', framealpha=0.7)

fig.tight_layout()
plt.show()