from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Define the directory containing your trial folders
base_dir = Path("Sensor/poking/1_good/good_breadboard")  # Adjust this path as needed
sensor_files = list(base_dir.glob("**/sensor.txt"))

print(f"Found {len(sensor_files)} total sensor files to evaluate.\n")

if not sensor_files:
    print("No sensor files found!")
    exit()

best_trial = None
min_error_metric = float('inf')
fixed_ratio_N = 0.01267076  # Fixed ratio as requested
n = 50                      # Window size for moving average

trial_summaries = []

# 1. PARSE AND EVALUATE EACH TRIAL USING YOUR EXACT CODE LOGIC
for file_path in sensor_files:
    number = []
    magnitude = []
    normal_force_sensor_value = []
    vectors_x = []
    vectors_y = []
    new_magnitude = []
    radio_N = []
    result_new_magnitude_plus = []
    
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
        
    # Calculate means and new magnitude (matching your script's loop structure)
    for idx in range(len(vectors_x)):
        mean_x = float(np.mean(vectors_x[:idx+1]))
        mean_y = float(np.mean(vectors_y[:idx+1])) if vectors_y[:idx+1] else 0.0
        
        new_vector = np.array([vectors_x[:idx+1], vectors_y[:idx+1]]).T
        new_vector[:, 0] -= mean_x
        new_vector[:, 1] -= mean_y
        
        tmp = [np.sqrt(np.square(vec[0]) + np.square(vec[1])) for vec in new_vector]
        new_magnitude.append(float(np.sum(tmp)))

    nf = float(np.sum(normal_force_sensor_value))
    nm = float(np.sum(magnitude))
    if nf != 0 and nm != 0:
        radio_N.append(nf / nm)

    # Scale values using the fixed ratio
    new_magnitude_plus = [i * fixed_ratio_N for i in magnitude]
    
    # Moving average filter matching your exact implementation
    for i in range(len(new_magnitude_plus)):
        window = new_magnitude_plus[i-n+1:i+1]
        result_new_magnitude_plus.append(sum(window) / n)
        
    # Calculate errors
    error_N = [result_new_magnitude_plus[i] - normal_force_sensor_value[i] for i in range(len(normal_force_sensor_value))]
    
    # Error metric (using RMSE to determine the smallest error)
    rmse = np.sqrt(np.mean(np.square(error_N)))
    trial_name = file_path.parent.name
    
    trial_summaries.append({
        "trial_name": trial_name,
        "path": file_path,
        "rmse": rmse,
        "number": number,
        "normal_force_sensor_value": normal_force_sensor_value,
        "result_new_magnitude_plus": result_new_magnitude_plus,
        "error_N": error_N
    })
    
    if rmse < min_error_metric:
        min_error_metric = rmse
        best_trial = trial_summaries[-1]

# Print ranking of all trials
trial_summaries.sort(key=lambda x: x["rmse"])
print("--- Trial Error Ranking (Lowest to Highest RMSE) ---")
for idx, t in enumerate(trial_summaries, 1):
    print(f"{idx}. [{t['trial_name']}] -> RMSE: {t['rmse']:.4f} N")

print(f"\n✨ Best Trial with Smallest Error: {best_trial['trial_name']} (RMSE: {best_trial['rmse']:.4f} N)\n")

# 2. PLOT THE BEST TRIAL USING YOUR EXACT PLOTTING CODE
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Normal Force comparison
axs[0].plot(best_trial["number"], best_trial["result_new_magnitude_plus"], linestyle='-', linewidth=2, markersize=2, label='Average')
axs[0].plot(best_trial["number"], best_trial["normal_force_sensor_value"], linestyle='-', linewidth=2, markersize=2, label='Sensor')
axs[0].legend(loc='upper left', framealpha=0.7, edgecolor='none')
axs[0].set_title(f'Normal Force vs Frame — Best Trial: {best_trial["trial_name"]}')
axs[0].set_xlabel('Frame (ticks)')
axs[0].set_ylabel('Normal Force (Newton)')
axs[0].grid(True)
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(200))

# Plot 2: Normal Force Error
axs[1].plot(best_trial["number"], best_trial["error_N"], marker='o', linestyle='-', color='g', linewidth=2, markersize=2)
axs[1].set_title('Normal Force Error vs Frame')
axs[1].set_xlabel('Frame (ticks)')
axs[1].set_ylabel('Error (N)')
axs[1].grid()

fig.tight_layout()
plt.show()