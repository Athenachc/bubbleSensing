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
error_threshold = 0.345
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
print(f"Refined best global ratio_N: {best_radio_N:.8f}")

if not valid_trial_data:
    print("No valid trials to plot!")
    exit()

# 4. Interpolate valid trials onto a common time grid for a clean conference aggregate plot
min_max_time = min([data["timestamp"][-1] for data in valid_trial_data])
common_time = np.linspace(0, min_max_time, 300)

interp_sensors = []
interp_calcs = []

for data in valid_trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    timestamps = data["timestamp"]
    
    calculated_nf = mag * best_radio_N
    
    if len(calculated_nf) >= n_window:
        calc_smooth = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        sensor_smooth = np.array(nf_sensor[n_window-1:])
        plot_time = np.array(timestamps[n_window-1:])
    else:
        calc_smooth = calculated_nf
        sensor_smooth = np.array(nf_sensor)
        plot_time = np.array(timestamps)
        
    # Interpolate onto common time axis
    s_interp = np.interp(common_time, plot_time, sensor_smooth)
    c_interp = np.interp(common_time, plot_time, calc_smooth)
    
    interp_sensors.append(s_interp)
    interp_calcs.append(c_interp)

mean_sensor = np.mean(interp_sensors, axis=0)
mean_calc = np.mean(interp_calcs, axis=0)
std_sensor = np.std(interp_sensors, axis=0)
std_calc = np.std(interp_calcs, axis=0)

# 5. Plot the compact aggregate gap graph for conference papers
fig, ax = plt.subplots(figsize=(8, 5))

# Plot mean Ground Truth and Calculation curves
ax.plot(common_time, mean_sensor, color='black', linewidth=2.2, label='Mean Force Sensor (Ground Truth)')
ax.plot(common_time, mean_calc, color='dodgerblue', linewidth=2, linestyle='--', label=f'Mean Calculated (radio_N = {best_radio_N:.5f})')

# Add labeled standard deviation envelopes to show trial repeatability in the legend
ax.fill_between(common_time, mean_sensor - std_sensor, mean_sensor + std_sensor, color='black', alpha=0.1, label='Sensor Trial Spread (±1σ)')
ax.fill_between(common_time, mean_calc - std_calc, mean_calc + std_calc, color='dodgerblue', alpha=0.1, label='Calculated Trial Spread (±1σ)')

# Highlight the primary "gap" between ground truth and calculation
ax.fill_between(common_time, mean_sensor, mean_calc, color='orange', alpha=0.4, label='Aggregate Discrepancy Gap')

ax.set_title('Aggregate Normal Force Progression & Discrepancy Gap Across Valid Trials', fontsize=11, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('Normal Force (N)', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper left', fontsize='small', framealpha=0.9)

fig.tight_layout()
plt.show()