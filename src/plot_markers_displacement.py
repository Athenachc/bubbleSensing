import math
import matplotlib.pyplot as plt

def get_drag_direction(dx, dy):
    # Compute angle in degrees (0° is Right, rotating clockwise where +y is Down)
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
                mag = math.sqrt(dx**2 + dy**2)
                direction = get_drag_direction(dx, dy)
                
                if mid not in marker_data:
                    marker_data[mid] = {'dx': [], 'dy': [], 'mag': [], 'direction': []}
                
                marker_data[mid]['dx'].append(dx)
                marker_data[mid]['dy'].append(dy)
                marker_data[mid]['mag'].append(mag)
                marker_data[mid]['direction'].append(direction)
                
    return frames, marker_data

def plot_all_data(file_path):
    frames, marker_data = process_sensor_data(file_path)
    
    # Create subplots: Magnitude, DX (Horizontal), DY (Vertical)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for mid, data in marker_data.items():
        ax1.plot(frames, data['mag'], label=f'Marker {mid}')
        ax2.plot(frames, data['dx'], label=f'Marker {mid}')
        ax3.plot(frames, data['dy'], label=f'Marker {mid}')
    
    # Formatting
    ax1.set_title('Displacement Magnitude (All Markers)')
    ax1.set_ylabel('Magnitude (px)')
    ax1.grid(True)
    
    ax2.set_title('Horizontal Displacement (dx: +Right, -Left)')
    ax2.set_ylabel('dx (px)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(True)
    
    ax3.set_title('Vertical Displacement (dy: +Down, -Up)')
    ax3.set_ylabel('dy (px)')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.grid(True)
    ax3.set_xlabel('Frame Number')
    
    plt.tight_layout()
    plt.show()

# Run the process
file_path = 'Sensor/interaction_separate/Trial_20260813_234118/sensor.txt'
plot_all_data(file_path)