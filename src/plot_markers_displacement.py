import math
import matplotlib.pyplot as plt
import numpy as np

# Threshold for determining active drag vs. stationary/pure push noise floor (in pixels)
DRAG_THRESHOLD = 1.5

INITIAL_POSITIONS = {
    0: ((128, 20), "Top-Left", 'o'),
    1: ((203, 20), "Top-Center", 's'),
    2: ((278, 20), "Top-Right", '^'),
    3: ((128, 95), "Middle-Left", 'v'),
    4: ((203, 95), "Middle-Center", '<'),
    5: ((278, 95), "Middle-Right", '>'),
    6: ((128, 170), "Bottom-Left", 'p'),
    7: ((203, 170), "Bottom-Center", '*'),
    8: ((278, 170), "Bottom-Right", 'X')
}

def get_drag_direction(dx, dy):
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
        
    if 22.5 <= angle < 67.5:
        return "DOWN-RIGHT"
    elif 67.5 <= angle < 112.5:
        return "DOWN"
    elif 112.5 <= angle < 157.5:
        return "DOWN-LEFT"
    elif 157.5 <= angle < 202.5:
        return "LEFT"
    elif 202.5 <= angle < 247.5:
        return "UP-LEFT"
    elif 247.5 <= angle < 292.5:
        return "UP"
    elif 292.5 <= angle < 337.5:
        return "UP-RIGHT"
    else:
        return "RIGHT"

def save_sensor_with_direction(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        # Write header if needed or handle existing comment headers
        first_line = True
        for line in infile:
            if not line.strip():
                continue
            
            if line.startswith('#'):
                outfile.write(line.strip() + ", Final_Direction\n")
                continue
            
            # If sensor.txt doesn't have a header, generate one on the first line
            if first_line:
                parts = [x.strip() for x in line.split(',')]
                num_coords = len(parts) - 2
                coord_headers = []
                for i in range(0, num_coords, 2):
                    coord_headers.extend([f"x_{i//2}", f"y_{i//2}"])
                header = ["Frame", "Flow_Magnitude"] + coord_headers + ["Final_Direction"]
                outfile.write(",".join(header) + "\n")
                first_line = False

            parts = [x.strip() for x in line.split(',')]
            frame_num = int(parts[0])
            coords = [float(x) for x in parts[2:]]
            
            dx_vals = coords[0::2]
            dy_vals = coords[1::2]
            mean_dx = sum(dx_vals) / len(dx_vals) if dx_vals else 0.0
            mean_dy = sum(dy_vals) / len(dy_vals) if dy_vals else 0.0
            magnitude = math.hypot(mean_dx, mean_dy)
            
            if magnitude < DRAG_THRESHOLD:
                final_direction = "NO-DRAG"
            else:
                final_direction = get_drag_direction(mean_dx, mean_dy)
                
            original_data_str = ",".join(parts)
            outfile.write(f"{original_data_str},{final_direction}\n")

def process_sensor_data(file_path):
    frames = []
    marker_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip() or 'Frame' in line or 'Flow_Magnitude' in line:
                continue
            
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3:
                continue
                
            try:
                frame_num = int(parts[0])
                # Coordinates are all elements between flow magnitude and the final text direction label
                coords = [float(x) for x in parts[2:-1]]
                final_direction = parts[-1]
            except ValueError:
                continue
            
            frames.append(frame_num)
            
            for i in range(0, len(coords) - 1, 2):
                mid = i // 2
                dx, dy = coords[i], coords[i+1]
                
                if mid not in marker_data:
                    marker_data[mid] = {'dx': [], 'dy': [], 'direction': []}
                
                marker_data[mid]['dx'].append(dx)
                marker_data[mid]['dy'].append(dy)
                marker_data[mid]['direction'].append(final_direction)
                
    return frames, marker_data

def plot_dx_dy_with_initial_state(file_path, fps=30):
    frames, marker_data = process_sensor_data(file_path)
    time_sec = [f / fps for f in frames]
    
    directions_list = ["NO-DRAG", "RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
    dir_to_idx = {d: i for i, d in enumerate(directions_list)}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    cmap = plt.get_cmap('tab10', len(directions_list))
    
    for mid, data in marker_data.items():
        pos_info, desc, marker_style = INITIAL_POSITIONS[mid]
        label_name = f"M{mid}: {desc} ({pos_info[0]}, {pos_info[1]})"
        numeric_dirs = [dir_to_idx.get(d, 0) for d in data['direction']]
        
        scatter1 = ax1.scatter(time_sec, data['dx'], c=numeric_dirs, cmap=cmap, vmin=-0.5, vmax=8.5, s=22, marker=marker_style, alpha=0.85, label=label_name)
        scatter2 = ax2.scatter(time_sec, data['dy'], c=numeric_dirs, cmap=cmap, vmin=-0.5, vmax=8.5, s=22, marker=marker_style, alpha=0.85, label=label_name)
    
    ax1.set_title('Horizontal Displacement (dx) Colored by Direction Classification')
    ax1.set_ylabel('dx (px)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(True)
    
    ax2.set_title('Vertical Displacement (dy) Colored by Direction Classification')
    ax2.set_ylabel('dy (px)')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.grid(True)
    ax2.set_xlabel('Time (seconds)')
    
    fig.subplots_adjust(right=0.71, top=0.92, bottom=0.1, hspace=0.3)
    
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, title="Initial States & Markers")
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(scatter1, cax=cbar_ax, ticks=range(len(directions_list)))
    cbar.set_ticklabels(directions_list)
    cbar.set_label('Direction Classification')
    
    plt.show()

input_file_path = 'Sensor/interaction_separate/Trial_20260813_233200/sensor.txt'
output_file_path = 'Sensor/interaction_separate/Trial_20260813_233200/sensor_with_direction.txt'

save_sensor_with_direction(input_file_path, output_file_path)
plot_dx_dy_with_initial_state(output_file_path, fps=30)