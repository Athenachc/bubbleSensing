from lib import find_marker
import cv2
import marker_dectection
import numpy as np
import sys
import setting
import time
import threading
import RS485

serial_port = '/dev/ttyUSB0'  #'/dev/ttyUSB0' or 'COM11'
reset = bytes([0x01, 0x10, 0x46, 0x08, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0xE8, 0x6A])
getdata1 = bytes([0x01, 0x04, 0x00, 0x0E, 0x00, 0x02, 0x10, 0x08])

calibrate = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'calibrate':
        calibrate = True


gelsight_version = 'Bnz'

cap = cv2.VideoCapture(4)

setting.init()
sensor = RS485.sensor_init(serial_port, reset)
RESCALE = setting.RESCALE

m = find_marker.Matching(
    N_=setting.N_, 
    M_=setting.M_, 
    fps_=setting.fps_, 
    x0_=setting.x0_, 
    y0_=setting.y0_, 
    dx_=setting.dx_, 
    dy_=setting.dy_)
    
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if gelsight_version == 'HSR':
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (215,215))
else:
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280//RESCALE,720//RESCALE))

# for i in range(30): ret, frame = cap.read()

for i in range(30): ret, frame = cap.read()
img = cv2.imread('calibresult.png')
frame = marker_dectection.init(img)
mask = marker_dectection.find_marker(frame)
mc = marker_dectection.marker_center(mask, frame)

cv2.imwrite('mask.png', mask)
cv2.imwrite('frame.png', frame)

file_center = open("center.txt", "w")
for i in mc:
    file_center.write(str(i) + ', ')
file_center.close()

file_sensor = open("Sensor/sensor.txt", "w")
num = 1

while(True):

    ret, frame = cap.read()
    if not(ret):
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
    
    if calibrate == False:
    
        tm = time.time()
        m.init(mc)

        # # matching
        m.run()
        print(time.time() - tm)
        
        
        flow = m.get_flow()

        # draw flow
        marker_dectection.draw_flow(frame, flow)

        vector = marker_dectection.get_flow_vector(flow, camera_center)
        RS485.sensor_send(sensor, getdata1)
        file_sensor.write(str(num) + ", " + str(RS485.sensor_read(sensor)) + ", " + str(marker_dectection.get_flow_magnitude(flow)) + ", " + str(vector[0]) + ", " + str(vector[1]) + ", " + str(marker_dectection.get_flow_center(flow, camera_center)) + '\n')
        cv2.imwrite("Sensor/frame" + str(num) + ".jpg", frame)
        num += 1
        
        
    mask_img = mask.astype(frame[0].dtype)
    mask_img = cv2.merge((mask_img, mask_img, mask_img))
    #RS485.sensor_send(sensor, getdata1)

    #cv2.imshow('raw',frame_raw)
    cv2.imshow('frame',frame)
    
    if calibrate:
        # Display the mask 
        cv2.imshow('mask',mask_img)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
file_sensor.close()
RS485.sensor_close(sensor)
cap.release()
out.release()
cv2.destroyAllWindows()
