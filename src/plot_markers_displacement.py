import math
import matplotlib.pyplot as plt
import numpy as np

# Initial positions, descriptions, and unique marker shapes for the 9 markers
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

def process_sensor_data(file_path):
    frames = []
    marker_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            data = [float(x.strip()) for x in line.split(',')]
            frame_num = int(data[0])
            coords = data[2:]  
            
            frames.append(frame_num)
            
            for i in range(0, len(coords), 2):
                mid = i // 2
                dx, dy = coords[i], coords[i+1]
                direction = get_drag_direction(dx, dy)
                
                if mid not in marker_data:
                    marker_data[mid] = {'dx': [], 'dy': [], 'direction': []}
                
                marker_data[mid]['dx'].append(dx)
                marker_data[mid]['dy'].append(dy)
                marker_data[mid]['direction'].append(direction)
                
    return frames, marker_data

def plot_dx_dy_with_initial_state(file_path, fps=30):
    frames, marker_data = process_sensor_data(file_path)
    time_sec = [f / fps for f in frames]
    
    directions_list = ["RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
    dir_to_idx = {d: i for i, d in enumerate(directions_list)}
    
    # Create exactly 2 subplots: DX and DY
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    cmap = plt.get_cmap('tab10', 8)
    
    # Plot each marker with its distinct shape and initial state description in the label
    for mid, data in marker_data.items():
        pos_info, desc, marker_style = INITIAL_POSITIONS[mid]
        label_name = f"M{mid}: {desc} ({pos_info[0]}, {pos_info[1]})"
        numeric_dirs = [dir_to_idx[d] for d in data['direction']]
        
        scatter1 = ax1.scatter(time_sec, data['dx'], c=numeric_dirs, cmap=cmap, vmin=-0.5, vmax=7.5, s=22, marker=marker_style, alpha=0.85, label=label_name)
        scatter2 = ax2.scatter(time_sec, data['dy'], c=numeric_dirs, cmap=cmap, vmin=-0.5, vmax=7.5, s=22, marker=marker_style, alpha=0.85, label=label_name)
    
    # Formatting DX plot
    ax1.set_title('Horizontal Displacement (dx) Colored by 8-Direction Classification')
    ax1.set_ylabel('dx (px)')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(True)
    
    # Formatting DY plot
    ax2.set_title('Vertical Displacement (dy) Colored by 8-Direction Classification')
    ax2.set_ylabel('dy (px)')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.grid(True)
    ax2.set_xlabel('Time (seconds)')
    
    # Adjust layout to leave space on the right for both the legend and the colorbar
    fig.subplots_adjust(right=0.71, top=0.92, bottom=0.1, hspace=0.3)
    
    # Place the initial states/markers legend safely outside the graphs on the right
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, title="Initial States & Markers")
    
    # Place the vertical colorbar for the 8 directions further to the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, ticks=range(8))
    cbar.set_ticklabels(directions_list)
    cbar.set_label('8-Direction Classification')
    
    plt.show()

file_path = 'Sensor/interaction_separate/Trial_20260813_234118/sensor.txt'
plot_dx_dy_with_initial_state(file_path, fps=30)