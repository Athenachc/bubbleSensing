"""
Marker Initial Position:
1. Top-Left (x1, y1) = (128, 20)
2. Top-Center (x2, y2) = (203, 20)
3. Top-Right (x3, y3) = (278, 20)
4. Middle-Left (x4, y4) = (128, 95)
5. Middle-Center (x5, y5) = (203, 95)
6. Middle-Right (x6, y6) = (278, 95 )
7. Bottom-Left (x7, y7) = (128 , 170)
8. Bottom-Center (x8, y8) = (203, 170)
9. Bottom-Right (x9, y9) = (278, 170)
"""

import math
import matplotlib.pyplot as plt

def process_sensor_data(file_path):
    frames = []
    # Store lists for each marker: {id: {'dx': [], 'dy': [], 'mag': []}}
    marker_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            data = [float(x.strip()) for x in line.split(',')]
            frame_num = int(data[0])
            coords = data[3:]
            
            frames.append(frame_num)
            
            for i in range(0, len(coords), 2):
                mid = i // 2
                dx, dy = coords[i], coords[i+1]
                mag = math.sqrt(dx**2 + dy**2)
                
                if mid not in marker_data:
                    marker_data[mid] = {'dx': [], 'dy': [], 'mag': []}
                
                marker_data[mid]['dx'].append(dx)
                marker_data[mid]['dy'].append(dy)
                marker_data[mid]['mag'].append(mag)
                
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
file_path = 'Sensor/Trial_20260713_222105/sensor.txt'
plot_all_data(file_path)