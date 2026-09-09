from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define the directory containing your 12 object folders
base_dir = Path("Sensor/poking/1_good")
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} object trial files to evaluate.\n")

if not sensor_files:
    print(f"No sensor files found in {base_dir}!")
    exit()

fixed_ratio_N = 0.01267076  # Fixed calibration ratio
n_window = 50                 # Moving average window size

trial_results = []

# 1. PARSE AND EVALUATE EACH OBJECT TRIAL
for file_path in sensor_files:
    number = []
    normal_force_sensor_value = []
    magnitude = []
    vectors_x = []
    vectors_y = []
    
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(", ", 5)
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
        
    # Calculate means and zero-center vectors (matching your core logic)
    mean_x = float(np.mean(vectors_x)) if vectors_x else 0.0
    mean_y = float(np.mean(vectors_y)) if vectors_y else 0.0
    
    new_vector = np.array([vectors_x, vectors_y]).T
    if len(new_vector) > 0:
        new_vector[:, 0] -= mean_x
        new_vector[:, 1] -= mean_y
        
    new_magnitude = [np.sqrt(np.square(vec[0]) + np.square(vec[1])) for vec in new_vector]
    
    # Apply fixed ratio_N
    new_magnitude_plus = [i * fixed_ratio_N for i in new_magnitude]
    
    # Moving average filter
    result_new_magnitude_plus = []
    for i in range(len(new_magnitude_plus)):
        window = new_magnitude_plus[max(0, i - n_window + 1):i + 1]
        result_new_magnitude_plus.append(sum(window) / len(window))
        
    sensor_arr = np.array(normal_force_sensor_value)
    detected_arr = np.array(result_new_magnitude_plus)
    
    # Error Metrics
    error_N = detected_arr - sensor_arr
    rmse = np.sqrt(np.mean(np.square(error_N)))
    max_abs_error = np.max(np.abs(error_N))
    
    true_peak = np.max(sensor_arr)
    detected_peak = np.max(detected_arr)
    peak_force_error_pct = (abs(detected_peak - true_peak) / true_peak * 100) if true_peak > 0 else 0.0
    
    # R-squared score
    ss_res = np.sum(np.square(error_N))
    ss_tot = np.sum(np.square(sensor_arr - np.mean(sensor_arr)))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    trial_results.append({
        "object_name": file_path.parent.name,
        "path": file_path,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "peak_force_error_pct": peak_force_error_pct,
        "r2": r2,
        "true_peak": true_peak,
        "detected_peak": detected_peak,
        "number": number,
        "sensor_force": sensor_arr,
        "detected_force": detected_arr
    })

# Sort trials by RMSE (lowest to highest error)
trial_results.sort(key=lambda x: x["rmse"])

# 2. PRINT EXECUTIVE SUMMARY TABLE
print("=" * 85)
print(f"{'Object / Trial Name':<30} | {'RMSE (N)':<10} | {'Max Err (N)':<12} | {'Peak Err (%)':<12} | {'R² Score':<8}")
print("-" * 85)
for t in trial_results:
    print(f"{t['object_name']:<30} | {t['rmse']:<10.4f} | {t['max_abs_error']:<12.4f} | {t['peak_force_error_pct']:<12.2f} | {t['r2']:<8.3f}")
print("=" * 85)

# 3. GENERATE COMPARATIVE VISUALIZATIONS (Parity Plot & RMSE Bar Chart)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Parity Plot (True Peak vs. Detected Peak across all 12 objects)
true_peaks = [t["true_peak"] for t in trial_results]
detected_peaks = [t["detected_peak"] for t in trial_results]
object_names = [t["object_name"] for t in trial_results]

axs[0].scatter(true_peaks, detected_peaks, color='dodgerblue', s=70, edgecolor='black', zorder=3)
for i, name in enumerate(object_names):
    axs[0].annotate(name, (true_peaks[i], detected_peaks[i]), fontsize=8, xytext=(5, 2), textcoords='offset points')

# Ideal unity line y = x
min_val = min(min(true_peaks), min(detected_peaks)) * 0.9
max_val = max(max(true_peaks), max(detected_peaks)) * 1.1
axs[0].plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1.5, label='Ideal Unity ($y=x$)')

axs[0].set_title('Peak Force Parity Plot (All 12 Objects)', fontsize=11, fontweight='bold')
axs[0].set_xlabel('Ground Truth Peak Force (N)', fontsize=10)
axs[0].set_ylabel('Detected Peak Force (N)', fontsize=10)
axs[0].grid(True, linestyle=':', alpha=0.7)
axs[0].legend(loc='upper left', fontsize=9)

# Plot B: RMSE Bar Chart Ranking
names_short = [t["object_name"] for t in trial_results]
rmse_vals = [t["rmse"] for t in trial_results]

bars = axs[1].barh(names_short, rmse_vals, color='crimson', alpha=0.8, edgecolor='black')
axs[1].set_title('Tracking Error Ranking (RMSE per Object)', fontsize=11, fontweight='bold')
axs[1].set_xlabel('RMSE (Newton)', fontsize=10)
axs[1].grid(True, linestyle=':', alpha=0.7, axis='x')
axs[1].invert_yaxis()  # Best performing (lowest RMSE) at the top

fig.tight_layout()
plt.show()