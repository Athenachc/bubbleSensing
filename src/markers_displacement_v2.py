# with RS485 reading
import cv2
import marker_dectection
import math
import numpy as np
import sys
import setting
import time
import threading
import RS485
import os
import datetime
from pathlib import Path
from lib import find_marker

# --- CONFIGURATION ---
serial_port = '/dev/ttyUSB0'  #'/dev/ttyUSB0' or 'COM11'
gelsight_version = 'Bnz'
reset = bytes([0x01, 0x10, 0x46, 0x08, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0xE8, 0x6A])
getdata1 = bytes([0x01, 0x04, 0x00, 0x0E, 0x00, 0x02, 0x10, 0x08])

def setup_trial_folder():
    """Creates a unique timestamped folder for each run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join('Sensor', f'Trial_{timestamp}')
    os.makedirs(folder, exist_ok=True)
    return folder

def main():
    # Setup & Initialization
    calibrate = 'calibrate' in sys.argv
    trial_folder = setup_trial_folder()
    print(f"Data saving to: {trial_folder}")

    setting.init()
    sensor = RS485.sensor_init(serial_port, reset)
    cap = cv2.VideoCapture(0)
    
    # VIDEO WRITER SETUP
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(trial_folder, 'output.mp4')
    if gelsight_version == 'HSR':
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (215, 215))
    else:
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (1280 // setting.RESCALE, 720 // setting.RESCALE))

    # CALIBRATION & WARM-UP
    for i in range(30): ret, frame = cap.read()
    img = cv2.imread('calibresult.png')
    frame = marker_dectection.init(img)
    mask = marker_dectection.find_marker(frame)
    mc = marker_dectection.marker_center(mask, frame)

    # SAVE INITIALIZATION FILES
    cv2.imwrite(os.path.join(trial_folder, 'mask.png'), mask)
    cv2.imwrite(os.path.join(trial_folder, 'frame.png'), frame)

    m = find_marker.Matching(
    N_=setting.N_, 
    M_=setting.M_, 
    fps_=setting.fps_, 
    x0_=setting.x0_, 
    y0_=setting.y0_, 
    dx_=setting.dx_, 
    dy_=setting.dy_)

    # Save the center points
    file_center = open(os.path.join(trial_folder, "center.txt"), "w")
    for i in mc:
        file_center.write(str(i) + ', ')
    file_center.close()

    # Open resources
    file_sensor = open(os.path.join(trial_folder, "sensor.txt"), "w")
    num = 1

    # Threshold for determining active drag vs. stationary/pure push noise floor (in pixels)
    DRAG_THRESHOLD = 1.5

    # Use 'try...finally' to ensure hardware releases even if the script crashes
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_raw = frame.copy()

            if gelsight_version == 'HSR':
                frame = marker_dectection.init_HSR(frame)
            else:
                frame = marker_dectection.init(frame)
            
            # find marker masks
            mask = marker_dectection.find_marker(frame)
            height, width, _ = frame.shape
            camera_center = (width/2, height/2)
            # find marker centers
            mc = marker_dectection.marker_center(mask, frame)

            drag_status = "Initializing..."

            if not calibrate:
                tm = time.time()
                m.init(mc)
                m.run()
                print(time.time() - tm)

                flow = m.get_flow()

                # draw flow vectors on frame
                marker_dectection.draw_flow(frame, flow)

                RS485.sensor_send(sensor, getdata1)
                sensor_val = RS485.sensor_read(sensor)
                
                # Get all marker flow vectors [x1, y1, x2, y2, ...]
                all_marker_data = marker_dectection.get_flow_vector(flow, (width/2, height/2))

                # --- 8-DIRECTIONAL REAL-TIME DRAG ANALYSIS ---
                if len(all_marker_data) > 0:
                    dx_vals = all_marker_data[0::2] # Even indices are dx
                    dy_vals = all_marker_data[1::2] # Odd indices are dy
                    
                    mean_dx = np.mean(dx_vals)
                    mean_dy = np.mean(dy_vals)
                    magnitude = math.sqrt(mean_dx**2 + mean_dy**2)

                    if magnitude < DRAG_THRESHOLD:
                        drag_status = "State: Pure Push / No Drag"
                    else:
                        # Calculate angle in degrees (-180 to 180)
                        # Note: OpenCV y-axis points down (+dy is down, -dy is up)
                        angle = math.degrees(math.atan2(mean_dy, mean_dx))
                        
                        # Shift angle range to 0 - 360 for easier sector mapping
                        if angle < 0:
                            angle += 360

                        # Map angle (segmented into 8 directions, 45 degrees each)
                        # 0° is Right, rotating clockwise:
                        if 22.5 <= angle < 67.5:
                            drag_status = f"Drag: DOWN-RIGHT"
                        elif 67.5 <= angle < 112.5:
                            drag_status = f"Drag: DOWN"
                        elif 112.5 <= angle < 157.5:
                            drag_status = f"Drag: DOWN-LEFT"
                        elif 157.5 <= angle < 202.5:
                            drag_status = f"Drag: LEFT"
                        elif 202.5 <= angle < 247.5:
                            drag_status = f"Drag: UP-LEFT"
                        elif 247.5 <= angle < 292.5:
                            drag_status = f"Drag: UP"
                        elif 292.5 <= angle < 337.5:
                            drag_status = f"Drag: UP-RIGHT"
                        else:
                            drag_status = f"Drag: RIGHT"
                
                # Convert the array to a comma-separated string for logging
                data_str = ", ".join([f"{val:.2f}" for val in all_marker_data])

                # Write everything to sensor.txt
                file_sensor.write(f"{num}, {sensor_val}, {marker_dectection.get_flow_magnitude(flow)}, {data_str}\n")

            # --- REAL-TIME VISUAL OVERLAY ---
            cv2.putText(frame, drag_status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, drag_status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if not calibrate:
                cv2.imwrite(os.path.join(trial_folder, f"frame{num}.jpg"), frame)
                print(f"Saved: frame{num}.jpg - {drag_status}")
                num += 1

            mask_img = mask.astype(frame[0].dtype)
            mask_img = cv2.merge((mask_img, mask_img, mask_img))

            cv2.imshow('frame', frame)

            if calibrate:
                cv2.imshow('mask', mask_img)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   

    finally:
        # Cleanup: Always runs
        file_sensor.close()
        RS485.sensor_close(sensor)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Trial complete. {num-1} frames saved.")

if __name__ == "__main__":
    main()