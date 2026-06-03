from lib import find_marker
import cv2
import marker_dectection
import numpy as np
import RS485
import sys
import setting
import time
import threading
import os
import csv
from datetime import datetime


# 1. Setup Logging Directory
log_dir = "Sensor"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Initialize Settings (ensure RESCALE is at the top level of setting.py)
setting.init()

# 3. RS485 Config for Channel 2
serial_port = '/dev/ttyUSB0'  
reset_cmd = bytes([0x01, 0x10, 0x46, 0x08, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0xE8, 0x6A])
get_data_cmd = bytes([0x01, 0x04, 0x00, 0x0E, 0x00, 0x02, 0x10, 0x08])

# 4. Hardware Initialization
cap = cv2.VideoCapture(5)
sensor = RS485.sensor_init(serial_port, reset_cmd)

# --- DYNAMIC SIZE DETECTION FOR VIDEO ---
ret, sample_frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    sys.exit()

# Get processed frame size to avoid VideoWriter "Failed to write frame" errors
sample_proc = marker_dectection.init(sample_frame)
frame_height, frame_width = sample_proc.shape[:2]
fps_save = 30.0 

m = find_marker.Matching(
    N_=setting.N_, M_=setting.M_, fps_=setting.fps_, 
    x0_=setting.x0_, y0_=setting.y0_, dx_=setting.dx_, dy_=setting.dy_)

# 5. Unique Filename Generation (Date & Time)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(log_dir, f"data_{timestamp}.csv")
video_path = os.path.join(log_dir, f"video_{timestamp}.mp4")

# CSV Setup
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Force_N', 'Mag_Pixels', 'Timestamp'])

# Video Setup (using mp4v codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps_save, (frame_width, frame_height))

num = 1
print(f"--- Experiment Started ---")
print(f"Saving to: {csv_path}")
print(f"Saving to: {video_path}")

try:
    while True:
        # --- RS485 READING ---
        RS485.sensor_send(sensor, get_data_cmd)
        force_val = RS485.sensor_read(sensor)

        # --- IMAGE PROCESSING ---
        ret, frame = cap.read()
        if not ret: break

        frame_proc = marker_dectection.init(frame)
        mask = marker_dectection.find_marker(frame_proc)
        mc = marker_dectection.marker_center(mask, frame_proc)
        
        m.init(mc)
        m.run()
        flow = m.get_flow()
        marker_dectection.draw_flow(frame_proc, flow)

        # --- DISPLACEMENT MAGNITUDE ---
        mag = 0
        if flow and len(flow) > 2:
            displacements = np.array(flow[2])
            if displacements.size > 0:
                mag = np.mean(np.sqrt(np.sum(displacements**2, axis=1)))

        # --- STATUS, LOGGING & VIDEO SAVE ---
        curr_time = time.time()
        print(f"F: {num} | Mag: {mag:.2f} | Force: {force_val}N", end='\r')
        
        csv_writer.writerow([num, force_val, round(mag, 4), round(curr_time, 4)])
        video_writer.write(frame_proc)
        
        if num % 20 == 0:
            csv_file.flush()

        cv2.imshow('Tracking', frame_proc)
        num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExperiment stopped manually.")

finally:
    # 6. Cleanup
    if 'csv_file' in locals():
        csv_file.close()
    if 'video_writer' in locals():
        video_writer.release() 
    if sensor:
        RS485.sensor_close(sensor)
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved {num} frames to {log_dir}")