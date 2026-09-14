import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

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

def process_sensor_data(file_path):
    df = pd.read_csv(file_path)
    df['Frame'] = range(1, len(df) + 1)
    
    frames = df['Frame'].tolist()
    marker_data = {}

    for _, row in df.iterrows():
        if str(row.iloc[0]).startswith('#'):
            continue
            
        try:
            coords = row.iloc[2:-1].values.astype(float)
        except ValueError:
            continue
        
        dx_vals = coords[0::2]
        dy_vals = coords[1::2]
        mean_dx = sum(dx_vals) / len(dx_vals) if len(dx_vals) > 0 else 0.0
        mean_dy = sum(dy_vals) / len(dy_vals) if len(dy_vals) > 0 else 0.0
        magnitude = math.hypot(mean_dx, mean_dy)
        
        if magnitude < DRAG_THRESHOLD:
            final_direction = "NO-DRAG"
        else:
            final_direction = get_drag_direction(mean_dx, mean_dy)
        
        for i in range(0, len(coords) - 1, 2):
            mid = i // 2
            dx = coords[i]
            # Invert dy so positive means UP
            dy = -coords[i+1]
            
            if mid not in marker_data:
                marker_data[mid] = {'dx': [], 'dy': [], 'direction': []}
            
            marker_data[mid]['dx'].append(dx)
            marker_data[mid]['dy'].append(dy)
            marker_data[mid]['direction'].append(final_direction)
                
    return frames, marker_data

def plot_dx_dy_with_initial_state(file_path, fps=30):
    frames, marker_data = process_sensor_data(file_path)
    time_sec = [f / fps for f in frames]
    
    platform_data = [
        {"t": 0.000,   "x": -0.228, "y": 0.228,  "Final_Direction": "NO-DRAG"},
        {"t": 6.767,   "x": -0.342, "y": 27.02,  "Final_Direction": "UP"},
        {"t": 8.233,   "x": 22.80,  "y": 22.69,  "Final_Direction": "UP-RIGHT"},
        {"t": 15.333,  "x": 31.47,  "y": 0.912,  "Final_Direction": "RIGHT"},
        {"t": 16.700,  "x": 26.45,  "y": -15.51, "Final_Direction": "DOWN-RIGHT"},
        {"t": 21.700,  "x": 0.228,  "y": -32.84, "Final_Direction": "DOWN"},
        {"t": 23.533,  "x": -23.72, "y": -14.02, "Final_Direction": "DOWN-LEFT"},
        {"t": 28.667,  "x": -29.87, "y": 1.482,  "Final_Direction": "LEFT"},
        {"t": 32.100,  "x": -19.84, "y": 23.60,  "Final_Direction": "UP-LEFT"}
    ]
    
    # Keep platform y as-is (positive means UP) so both platform and markers follow the same convention
    dir_to_platform_xy = {item["Final_Direction"]: (item["x"], item["y"]) for item in platform_data}
    
    directions_list = ["NO-DRAG", "RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
    dir_to_idx = {d: i for i, d in enumerate(directions_list)}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    ax1_plat = ax1.twinx()
    ax2_plat = ax2.twinx()
    
    cmap = plt.get_cmap('tab10', len(directions_list))
    norm = plt.Normalize(vmin=-0.5, vmax=8.5)
    
    scatter1 = None
    for mid, data in marker_data.items():
        if mid not in INITIAL_POSITIONS:
            continue
        pos_info, desc, marker_style = INITIAL_POSITIONS[mid]
        label_name = f"M{mid}: {desc} ({pos_info[0]}, {pos_info[1]})"
        numeric_dirs = [dir_to_idx.get(d, 0) for d in data['direction']]
        
        edge_colors = cmap(norm(numeric_dirs))
        
        scatter1 = ax1.scatter(time_sec, data['dx'], facecolors='none', edgecolors=edge_colors, linewidths=0.8, s=75, marker=marker_style, alpha=0.95, label=label_name)
        ax2.scatter(time_sec, data['dy'], facecolors='none', edgecolors=edge_colors, linewidths=0.8, s=75, marker=marker_style, alpha=0.95, label=label_name)
    
    plat_dx_vals = []
    plat_dy_vals = []
    sample_directions = marker_data[0]['direction'] if 0 in marker_data else ["NO-DRAG"] * len(frames)
    
    for d in sample_directions:
        if d in dir_to_platform_xy:
            px, py = dir_to_platform_xy[d]
        else:
            px, py = (0.0, 0.0)
        plat_dx_vals.append(px)
        plat_dy_vals.append(py)
        
    line_plat_x = ax1_plat.plot(time_sec, plat_dx_vals, color='magenta', linestyle='-', linewidth=2.0, alpha=0.8, label='Platform X')
    line_plat_y = ax2_plat.plot(time_sec, plat_dy_vals, color='purple', linestyle='-', linewidth=2.0, alpha=0.8, label='Platform Y')
    
    ax1.set_title('Horizontal Displacement (dx) & Platform Trajectory Colored by Direction', fontsize=12)
    ax1.set_ylabel('Markers dx (px)')
    ax1_plat.set_ylabel('Platform X (px)', color='magenta')
    ax1_plat.tick_params(axis='y', labelcolor='magenta')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    ax2.set_title('Vertical Displacement (dy) & Platform Trajectory Colored by Direction (Positive = Up)', fontsize=12)
    ax2.set_ylabel('Markers dy (px)')
    ax2_plat.set_ylabel('Platform Y (px)', color='purple')
    ax2_plat.tick_params(axis='y', labelcolor='purple')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    
    fig.subplots_adjust(right=0.68, top=0.92, bottom=0.1, hspace=0.3)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_plat1, labels_plat1 = ax1_plat.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_plat1, labels_1 + labels_plat1, loc='center left', bbox_to_anchor=(1.12, 0.5), fontsize=8, title="Markers & Platform", framealpha=0.9)
    
    if scatter1 is not None:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(len(directions_list)))
        cbar.set_ticklabels(directions_list)
        cbar.set_label('Direction Classification')
    
    plt.show()

if __name__ == "__main__":
    file_name = 'Sensor/interaction_separate/sensor_with_direction_selected.txt'
    plot_dx_dy_with_initial_state(file_name, fps=30)