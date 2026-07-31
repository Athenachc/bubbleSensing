import cv2
import math
import numpy as np
import setting

def init(frame):
    RESCALE = setting.RESCALE
    return cv2.resize(frame, (0, 0), fx=1.0/RESCALE, fy=1.0/RESCALE)

def find_marker(frame):
    RESCALE = setting.RESCALE
    
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    h, w = mask.shape
    
    '''
    bottom = mask[int(h*0.9):h, :]
    clahe = cv2.createCLAHE(clipLimit=1.36, tileGridSize=(7,7))
    bottom_enhanced = clahe.apply(bottom)
    mask[int(h*0.9):h, :] = bottom_enhanced
    '''
    
    mask = cv2.medianBlur(mask, 23) #cencal bubbles
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 29, 11)
    
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    '''
    
    return mask


# def marker_center(mask, frame):
#     RESCALE = setting.RESCALE
    
#     areaThresh1 = 300/RESCALE**2
#     areaThresh2 = 10000/RESCALE**2
#     MarkerCenter = []

#     contours=cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours[0])<9:  # if too little markers, then give up
#         print("Too less markers detected: ", len(contours))
#         return MarkerCenter

#     for contour in contours[0]:
#         x,y,w,h = cv2.boundingRect(contour)
#         AreaCount=cv2.contourArea(contour)
#         # print(AreaCount)
#         if AreaCount>areaThresh1 and AreaCount<areaThresh2 and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1 and x>30 and x<275: 
#             t=cv2.moments(contour)
#             # print("moments", t)
#             # MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
#             mc = [t['m10']/t['m00'], t['m01']/t['m00']]
#             # if t['mu11'] < -100: continue
#             MarkerCenter.append(mc)
#             #print(mc)
#             cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

#     # 0:x 1:y
#     return MarkerCenter

def marker_center(mask, frame):
    RESCALE = setting.RESCALE
    
    areaThresh1 = 300 / RESCALE**2
    areaThresh2 = 5000 / RESCALE**2  # Lowered upper bound to reject large lighting blobs/objects
    MarkerCenter = []

    # Optional: Clean up small specs and holes caused by lighting/colors
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_cleaned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 9:
        print("Too less markers detected: ", len(contours))
        return MarkerCenter

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        AreaCount = cv2.contourArea(contour)
        
        # Calculate circularity or aspect ratio to ensure they are dot-like markers
        aspect_ratio = np.max([w, h]) * 1.0 / np.min([w, h])
        
        # Filter by area, strict aspect ratio, and ignore blobs too close to the image border (removes edge light noise)
        img_h, img_w = mask.shape
        if (areaThresh1 < AreaCount < areaThresh2) and (aspect_ratio < 1.4) and (20 < x < img_w - 20) and (20 < y < img_h - 20):
            valid_contours.append((AreaCount, contour))

    # If we found more than 9, sort by area (or closeness to expected size) and take the best 9
    # (Assuming your grid is strictly 3x3 = 9 markers)
    if len(valid_contours) >= 9:
        # Sort descending by area, or you can sort by how close they are to an expected median marker size
        valid_contours = sorted(valid_contours, key=lambda item: item[0], reverse=True)
        valid_contours = valid_contours[:9] # Keep only the top 9 candidates

    for AreaCount, contour in valid_contours:
        t = cv2.moments(contour)
        if t['m00'] == 0:
            continue
        mc = [t['m10'] / t['m00'], t['m01'] / t['m00']]
        MarkerCenter.append(mc)
        cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, (0, 0, 255), 2, 6)

    return MarkerCenter

def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.2)


def get_flow_magnitude(flow):
    Ox, Oy, Cx, Cy, Occupied = flow
    magnitude = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j]), int(Cy[i][j]))
            magnitude += math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    return round(magnitude, 3)
    
def get_flow_vector(flow, center):
    Ox, Oy, Cx, Cy, Occupied = flow
    # summed_vector = np.array([0, 0])
    # Create an empty list to store individual vectors
    all_vectors = []
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j]), int(Cy[i][j]))
            vector = np.array([round(pt2[0] - pt1[0], 3), round(pt2[1] - pt1[1], 3)])
            all_vectors.append(vector)
    # Flatten the list of vectors into a single 1D array (e.g., [x1, y1, x2, y2, ... x9, y9])
    return np.hstack(all_vectors)

def get_flow_center(flow, center):
    Ox, Oy, Cx, Cy, Occupied = flow
    magnitude_center = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            magnitude_center += math.sqrt((pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2)
    return round(magnitude_center, 3)


def warp_perspective(img):

    TOPLEFT = (175,230)
    TOPRIGHT = (380,225)
    BOTTOMLEFT = (10,410)
    BOTTOMRIGHT = (530,400)

    WARP_W = 215
    WARP_H = 215

    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
    points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])

    matrix=cv2.getPerspectiveTransform(points1,points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W,WARP_H))

    return result


def init_HSR(img):
    DIM=(640, 480)
    img = cv2.resize(img, DIM)

    K=np.array([[225.57469247811056, 0.0, 280.0069549918857], [0.0, 221.40607131318117, 294.82435570493794], [0.0, 0.0, 1.0]])
    D=np.array([[0.7302503082668154], [-0.18910060205317372], [-0.23997727800712282], [0.13938490908400802]])
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warp_perspective(undistorted_img)
