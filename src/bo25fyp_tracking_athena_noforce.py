import cv2
import marker_dectection
import numpy as np
import sys
import setting
import time
import threading
import os
import datetime
from pathlib import Path
from lib import find_marker

# --- CONFIGURATION ---
gelsight_version = 'Bnz'

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

            if not calibrate:
                tm = time.time()
                m.init(mc)
                m.run()
                print(time.time() - tm)

                flow = m.get_flow()

                # draw flow
                marker_dectection.draw_flow(frame, flow)

                # Call your modified function
                all_marker_data = marker_dectection.get_flow_vector(flow, (frame.shape[1]/2, frame.shape[0]/2))

                # Convert the array to a comma-separated string with 2 decimal places
                data_str = ", ".join([f"{val:.2f}" for val in all_marker_data])

                # Write format: frame_number, None, flow_magnitude, x1, y1, x2, y2, ...
                flow_mag = marker_dectection.get_flow_magnitude(flow)
                file_sensor.write(f"{num}, None, {flow_mag:.2f}, {data_str}\n")
                
                cv2.imwrite(os.path.join(trial_folder, f"frame{num}.jpg"), frame)
                print(f"Saved: frame{num}.jpg")
                num += 1

            # Code below this point runs EVERY frame, whether you are calibrating or recording
            mask_img = mask.astype(frame[0].dtype)
            mask_img = cv2.merge((mask_img, mask_img, mask_img))

            cv2.imshow('frame', frame)

            if calibrate:
                cv2.imshow('mask',mask_img)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   

    finally:
        # Cleanup: Always runs
        file_sensor.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Trial complete. {num-1} frames saved.")

if __name__ == "__main__":
    main()