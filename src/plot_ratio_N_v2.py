from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing your trial folders
base_dir = Path("Sensor/5mm_2.5ml_v3")
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} total sensor files to evaluate.")

trial_data = []
all_ratios = []
fps = 30.0  # Recording frame rate

# 1. Parse all trials and convert frame numbers to timestamps
for file_path in sensor_files:
    timestamps = []
    nf_values = []
    magnitudes = []
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ", 5)
            if len(parts) >= 5:
                try:
                    frame_num = int(parts[0].strip())
                    # Calculate timestamp in seconds starting from 0.0 for the first frame
                    t = (frame_num - 1) / fps
                    timestamps.append(t)
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
            "timestamp": timestamps,
            "normal_force": nf_values,
            "magnitude": magnitudes
        })

# Initial rough global ratio to test errors
initial_radio_N = np.mean(all_ratios)

# 2. Filter out trials where the maximum absolute error exceeds 0.3 N
error_threshold = 0.3
n_window = 50
valid_trial_data = []
filtered_ratios = []

for data in trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    timestamps = data["timestamp"]
    
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

# 4. Plot each valid trial with timestamps on the X-axis
n_valid = len(valid_trial_data)
if n_valid == 0:
    print("No valid trials to plot!")
    exit()

ncols = 2 if n_valid > 1 else 1
nrows = (n_valid + ncols - 1) // ncols

fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=True)
if n_valid == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i, data in enumerate(valid_trial_data):
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    timestamps = data["timestamp"]
    
    calculated_nf = mag * best_radio_N
    
    if len(calculated_nf) >= n_window:
        calc_smooth = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        sensor_smooth = np.array(nf_sensor[n_window-1:])
        plot_time = timestamps[n_window-1:]
    else:
        calc_smooth = calculated_nf
        sensor_smooth = np.array(nf_sensor)
        plot_time = timestamps
        
    trial_name = data["path"].parent.name
    
    # Plot Ground Truth (Force Sensor) and Calculation using Time (seconds)
    axs[i].plot(plot_time, sensor_smooth, color='black', linewidth=2, label='Force Sensor (Ground Truth)')
    axs[i].plot(plot_time, calc_smooth, color='dodgerblue', linewidth=1.8, linestyle='--', label='Calculated')
    
    # Shade the "gap" between ground truth and calculation
    axs[i].fill_between(plot_time, sensor_smooth, calc_smooth, color='orange', alpha=0.35, label='Discrepancy Gap')
    
    axs[i].set_title(f'Trial: {trial_name}', fontsize=11, fontweight='bold')
    axs[i].set_xlabel('Time (seconds)')
    axs[i].set_ylabel('Normal Force (N)')
    axs[i].grid(True, linestyle=':', alpha=0.7)
    axs[i].legend(loc='upper left', fontsize='small', framealpha=0.8)

# Hide any empty subplots if grid has spare slots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

fig.suptitle(f'Individual Trial Progression & Discrepancy Gaps over Time (Unified radio_N = {best_radio_N:.5f})', fontsize=14, y=0.98)
fig.tight_layout()
plt.show()