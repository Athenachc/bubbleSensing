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
file_path = 'sensor.txt'
data = process_sensor_data(file_path)

# Display results
for entry in data:
    print(f"--- Frame {entry['frame']} ---")
    for m in entry['markers']:
        print(f"Marker {m['marker_id']}: Mag={m['magnitude_px']}px, Dir={m['direction_deg']}°")