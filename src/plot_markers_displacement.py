import math
import matplotlib.pyplot as plt

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

def plot_dx_dy_with_directions(file_path, fps=30):
    frames, marker_data = process_sensor_data(file_path)
    
    # Convert frame numbers to time (seconds)
    time_sec = [f / fps for f in frames]
    
    direction_map = {
        "RIGHT": 0, "UP-RIGHT": 1, "UP": 2, "UP-LEFT": 3,
        "LEFT": 4, "DOWN-LEFT": 5, "DOWN": 6, "DOWN-RIGHT": 7
    }
    dir_names = list(direction_map.keys())
    
    # Create subplots: DX, DY, and 8-Direction Classification
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for mid, data in marker_data.items():
        ax1.plot(time_sec, data['dx'], label=f'Marker {mid}')
        ax2.plot(time_sec, data['dy'], label=f'Marker {mid}')
        
        numeric_dirs = [direction_map[d] for d in data['direction']]
        ax3.scatter(time_sec, numeric_dirs, s=10, label=f'Marker {mid}', alpha=0.7)
    
    # Formatting Subplot 1 (Horizontal Displacement)
    ax1.set_title('Horizontal Displacement (dx: +Right, -Left)')
    ax1.set_ylabel('dx (px)')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.grid(True)
    ax1.legend(loc='upper right', ncol=3, fontsize=8)
    
    # Formatting Subplot 2 (Vertical Displacement)
    ax2.set_title('Vertical Displacement (dy: +Down, -Up)')
    ax2.set_ylabel('dy (px)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(True)
    
    # Formatting Subplot 3 (8-Directions over time)
    ax3.set_title('Classified 8-Direction Movement Over Time')
    ax3.set_ylabel('Direction')
    ax3.set_yticks(list(direction_map.values()))
    ax3.set_yticklabels(dir_names)
    ax3.set_ylim(-0.5, 7.5)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()

# Run the process with 30 FPS
file_path = 'Sensor/interaction_separate/Trial_20260813_234118/sensor.txt'
plot_dx_dy_with_directions(file_path, fps=30)