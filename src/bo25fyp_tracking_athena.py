import cv2
import marker_dectection
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
    cap = cv2.VideoCapture(4)
    # VIDEO WRITER SETUP
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(trial_folder, 'output.mp4')
    if gelsight_version == 'HSR':
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (215, 215))
    else:
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (1280 // setting.RESCALE, 720 // setting.RESCALE))

    # CALIBRATION & WARM-UP (The lines you were missing)
    for i in range(30): ret, frame = cap.read()
    img = cv2.imread('calibresult.png')
    frame = marker_dectection.init(img)
    mask = marker_dectection.find_marker(frame)
    mc = marker_dectection.marker_center(mask, frame)

    # SAVE INITIALIZATION FILES (If you still want these saved per trial)
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

                vector = marker_dectection.get_flow_vector(flow, camera_center)
                RS485.sensor_send(sensor, getdata1)
                # file_sensor.write(str(num) + ", " + str(RS485.sensor_read(sensor)) + ", " + str(marker_dectection.get_flow_magnitude(flow)) + ", " + str(vector[0]) + ", " + str(vector[1]) + ", " + str(marker_dectection.get_flow_center(flow, camera_center)) + '\n')
                sensor_val = RS485.sensor_read(sensor)
                # Call your modified function
                all_marker_data = marker_dectection.get_flow_vector(flow, (frame.shape[1]/2, frame.shape[0]/2))

                # Convert the array to a comma-separated string
                # This turns [x1, y1, x2, y2...] into "x1, y1, x2, y2..."
                data_str = ", ".join([f"{val:.2f}" for val in all_marker_data])

                # Write everything in one line
                file_sensor.write(f"{num}, {sensor_val}, {marker_dectection.get_flow_magnitude(flow)}, {data_str}\n")
                # file_sensor.write(f"{num}, {sensor_val}, {marker_dectection.get_flow_magnitude(flow)}, {marker_dectection.get_flow_vector(flow, (frame.shape[1]/2, frame.shape[0]/2))}\n")
                # file_sensor.write(f"{num}, {sensor_val}, {marker_dectection.get_flow_magnitude(flow)}, {vector[0]}, {vector[1]}, {marker_dectection.get_flow_center(flow, (frame.shape[1]/2, frame.shape[0]/2))}\n")
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
        # 3. Cleanup: Always runs
        file_sensor.close()
        RS485.sensor_close(sensor)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Trial complete. {num-1} frames saved.")

if __name__ == "__main__":
    main()

# calibrate = False

# if len(sys.argv) > 1:
#     if sys.argv[1] == 'calibrate':
#         calibrate = True

# # Create unique trial folder based on current timestamp
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# trial_folder = os.path.join('Sensor', f'Trial_{timestamp}')
# os.makedirs(trial_folder, exist_ok=True)
# print(f"Data will be saved to: {trial_folder}")

# cap = cv2.VideoCapture(4)
# setting.init()
# sensor = RS485.sensor_init(serial_port, reset)
# RESCALE = setting.RESCALE

# m = find_marker.Matching(
#     N_=setting.N_, 
#     M_=setting.M_, 
#     fps_=setting.fps_, 
#     x0_=setting.x0_, 
#     y0_=setting.y0_, 
#     dx_=setting.dx_, 
#     dy_=setting.dy_)
    
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# if gelsight_version == 'HSR':
#     out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (215,215))
# else:
#     out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280//RESCALE,720//RESCALE))

# # for i in range(30): ret, frame = cap.read()

# for i in range(30): ret, frame = cap.read()
# img = cv2.imread('calibresult.png')
# frame = marker_dectection.init(img)
# mask = marker_dectection.find_marker(frame)
# mc = marker_dectection.marker_center(mask, frame)

# cv2.imwrite('mask.png', mask)
# cv2.imwrite('frame.png', frame)
# cv2.imwrite(os.path.join(trial_folder, 'mask.png'), mask)
# cv2.imwrite(os.path.join(trial_folder, 'frame.png'), frame)

# file_center = open("center.txt", "w")
# file_center = open(os.path.join(trial_folder, "center.txt"), "w")
# for i in mc:
#     file_center.write(str(i) + ', ')
# file_center.close()

# file_sensor = open("Sensor/sensor.txt", "w")
# Open sensor file inside the trial folder
# file_sensor = open(os.path.join(trial_folder, "sensor.txt"), "w")
# num = 1

# while(True):

    # ret, frame = cap.read()
    # if not(ret):
    #     break

    # frame_raw = frame.copy()
    
    # if gelsight_version == 'HSR':
    #     frame = marker_dectection.init_HSR(frame)
    # else:
    #     frame = marker_dectection.init(frame)

    # # find marker masks
    # mask = marker_dectection.find_marker(frame)
    # height, width, _ = frame.shape
    # camera_center = (width/2, height/2)

    # # find marker centers
    # mc = marker_dectection.marker_center(mask, frame)
    
    # if calibrate == False:
    
        # tm = time.time()
        # m.init(mc)

        # # # matching
        # m.run()
        # print(time.time() - tm)
        
        
        # flow = m.get_flow()

        # # draw flow
        # marker_dectection.draw_flow(frame, flow)

        # vector = marker_dectection.get_flow_vector(flow, camera_center)
        # RS485.sensor_send(sensor, getdata1)
        # # file_sensor.write(str(num) + ", " + str(RS485.sensor_read(sensor)) + ", " + str(marker_dectection.get_flow_magnitude(flow)) + ", " + str(vector[0]) + ", " + str(vector[1]) + ", " + str(marker_dectection.get_flow_center(flow, camera_center)) + '\n')
        # sensor_val = RS485.sensor_read(sensor)
        
        # # Save sensor data
        # file_sensor.write(f"{num}, {sensor_val}, {marker_dectection.get_flow_magnitude(flow)}, {vector[0]}, {vector[1]}, {marker_dectection.get_flow_center(flow, camera_center)}\n")
        # # cv2.imwrite("Sensor/frame" + str(num) + ".jpg", frame)
        # cv2.imwrite(os.path.join(trial_folder, f"frame{num}.jpg"), frame)
        # print(f"Saved: frame{num}.jpg")
        # num += 1
        
        
    # mask_img = mask.astype(frame[0].dtype)
    # mask_img = cv2.merge((mask_img, mask_img, mask_img))
    #RS485.sensor_send(sensor, getdata1)

    #cv2.imshow('raw',frame_raw)
    # cv2.imshow('frame',frame)
    
    # if calibrate:
    #     # Display the mask 
    #     cv2.imshow('mask',mask_img)

    # out.write(frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# # Cleanup
# file_sensor.close()
# RS485.sensor_close(sensor)
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# print(f"Trial complete. {num-1} frames saved.")
