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
import pandas as pd
import matplotlib.pyplot as plt

def process_sensor_data(file_path):
    # Defining column names: frame, sensor_val, flow_mag, then marker pairs
    # Since the number of markers might vary, we read the file line by line
    results = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            data = [float(x.strip()) for x in line.split(',')]
            frame_num = int(data[0])
            coords = data[3:] # Everything after index 2
            
            frame_summary = {
                "frame": frame_num,
                "markers": []
            }
            
            # Iterate through coordinates in pairs
            for i in range(0, len(coords), 2):
                dx = coords[i]
                dy = coords[i+1]
                
                # Magnitude calculation
                magnitude = math.sqrt(dx**2 + dy**2)
                
                # Direction calculation (in degrees)
                # math.atan2(y, x) handles the signs correctly
                direction = math.degrees(math.atan2(dy, dx))
                
                frame_summary["markers"].append({
                    "marker_id": i // 2,
                    "dx": dx,
                    "dy": dy,
                    "magnitude_px": round(magnitude, 4),
                    "direction_deg": round(direction, 2)
                })
            results.append(frame_summary)
    return results

# --- Main Execution ---
file_path = 'Sensor/Trial_20260713_222105/sensor.txt'
data = process_sensor_data(file_path)

# Display results
for entry in data:
    print(f"--- Frame {entry['frame']} ---")
    for m in entry['markers']:
        print(f"Marker {m['marker_id']}: Mag={m['magnitude_px']}px, Dir={m['direction_deg']}°")

def process_and_plot(file_path):
    frames = []
    # Using a dictionary to store lists of magnitudes for each marker_id
    # {marker_id: [mag_f1, mag_f2, ...]}
    marker_magnitudes = {}

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
                
                if mid not in marker_magnitudes:
                    marker_magnitudes[mid] = []
                marker_magnitudes[mid].append(mag)

    # Plotting
    plt.figure(figsize=(10, 6))
    for mid, magnitudes in marker_magnitudes.items():
        plt.plot(frames, magnitudes, marker='o', label=f'Marker {mid}')
    
    plt.title('Marker Displacement Magnitude per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Displacement Magnitude (pixels)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the process
process_and_plot('Sensor/Trial_20260713_222105/sensor.txt')