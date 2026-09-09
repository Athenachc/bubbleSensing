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

# 2. Filter out trials where the maximum absolute error exceeds 0.345 N
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
print(f"Refined best global ratio_N: {best_radio_N:.8f}")

if not valid_trial_data:
    print("No valid trials to plot!")
    exit()

# 4. Interpolate valid trials onto a common time grid (in seconds) for clean aggregation
min_max_frames = min([len(data["number"]) - n_window + 1 for data in valid_trial_data])
min_max_duration = (min_max_frames - 1) / fps
common_time = np.linspace(0, min_max_duration, 300)

interp_sensors = []
interp_calcs = []
interp_errors = []

for data in valid_trial_data:
    mag = np.array(data["magnitude"])
    nf_sensor = np.array(data["normal_force"])
    numbers = data["number"]
    
    calculated_nf = mag * best_radio_N
    
    if len(calculated_nf) >= n_window:
        calc_smooth = np.convolve(calculated_nf, np.ones(n_window)/n_window, mode='valid')
        sensor_smooth = np.array(nf_sensor[n_window-1:])
        plot_time = np.arange(len(calc_smooth)) / fps
    else:
        calc_smooth = calculated_nf
        sensor_smooth = np.array(nf_sensor)
        plot_time = np.arange(len(calc_smooth)) / fps
        
    error_N = calc_smooth - sensor_smooth
    
    # Interpolate onto shared time grid
    s_interp = np.interp(common_time, plot_time, sensor_smooth)
    c_interp = np.interp(common_time, plot_time, calc_smooth)
    e_interp = np.interp(common_time, plot_time, error_N)
    
    interp_sensors.append(s_interp)
    interp_calcs.append(c_interp)
    interp_errors.append(e_interp)

mean_sensor = np.mean(interp_sensors, axis=0)
mean_calc = np.mean(interp_calcs, axis=0)
std_calc = np.std(interp_calcs, axis=0)

mean_error = np.mean(interp_errors, axis=0)
std_error = np.std(interp_errors, axis=0)

# 5. Plot Clean Aggregate Curves against Time (seconds)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top Panel: Normal Force Comparison
axs[0].plot(common_time, mean_sensor, color='black', linewidth=2, label='Force Sensor (Ground Truth)')
axs[0].plot(common_time, mean_calc, color='dodgerblue', linewidth=2, linestyle='--', label='Calculated (Mean)')
axs[0].fill_between(common_time, mean_calc - std_calc, mean_calc + std_calc, color='dodgerblue', alpha=0.2, label='±1σ Trial Spread')
axs[0].set_title(f'Aggregate Normal Force Comparison (Filtered {len(valid_trial_data)} Trials, radio_N = {best_radio_N:.5f})', fontsize=11, fontweight='bold')
axs[0].set_ylabel('Normal Force (N)')
axs[0].grid(True, linestyle=':', alpha=0.7)
axs[0].legend(loc='upper left', fontsize='small', framealpha=0.9)

# Bottom Panel: Aggregate Error Trend with Shaded Standard Deviation Band
axs[1].plot(common_time, mean_error, color='crimson', linewidth=2, label='Mean Error (Detected - Measured)')
axs[1].fill_between(common_time, mean_error - std_error, mean_error + std_error, color='crimson', alpha=0.2, label='±1σ Error Spread')
axs[1].axhline(0, color='black', linestyle='-', linewidth=1)
# axs[1].axhline(error_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'+{error_threshold}N Threshold')
# axs[1].axhline(-error_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'-{error_threshold}N Threshold')

axs[1].set_title('Aggregate Normal Force Error vs Time', fontsize=11, fontweight='bold')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Error (N)')
axs[1].grid(True, linestyle=':', alpha=0.7)
axs[1].legend(loc='upper left', fontsize='small', framealpha=0.9)

fig.tight_layout()
plt.show()