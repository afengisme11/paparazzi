import numpy as np
# import extract_information_flow_field as OF
import matplotlib.pyplot as plt
import os
import re
import cv2
from loadCamCoefficients import load_coefficients


# Change the datasets directory locally
image_dir_name = os.getcwd() + '/Datasets/cyberzoo_poles/20190121-135009/'
image_type = 'jpg'
image_names = []
# Sort image names
for file in os.listdir(image_dir_name):
    if file.endswith(image_type):
        image_names.append(image_dir_name + file)
image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(image_names[88])
random_img = cv2.imread(image_names[0])
# Load camera calibration coefficients
mtx, dist = load_coefficients(os.getcwd()+'/camera_coefficients.yaml')
h, w = random_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
# Read images
prev_image = cv2.imread(image_names[285])
image = cv2.imread(image_names[289])
# Undistort images
prev_image = cv2.undistort(prev_image, mtx, dist, None, newcameramtx)
image = cv2.undistort(image, mtx, dist, None, newcameramtx)


lower_pole = np.array([0, 104, 122], dtype="uint8")
upper_pole = np.array([179, 255, 255], dtype="uint8")
lower_floor = np.array([0, 0, 0], dtype="uint8")
upper_floor = np.array([121, 150, 95], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

for i, img_file in enumerate(image_names):
    image_raw = cv2.imread(img_file)
    # Undistort images
    image_undist = cv2.undistort(image_raw, mtx, dist, None, newcameramtx)
    image = cv2.cvtColor(image_undist, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, lower_pole, upper_pole)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    max_area = 0
    how_many_poles = 0
    index_max_area = 10
    for j, c in enumerate(cnts):
        contour_area = cv2.contourArea(c)
        if contour_area > 1000:
            how_many_poles = how_many_poles + 1
            if contour_area > max_area:
                # The one with the largest area is the closest
                max_area = contour_area
                index_max_area = j
            cv2.drawContours(image_undist, [c], 0, (0,0,0), 2)

    # ************** Strategy *****************
    # This only covers the pole avoidance, floor detection to avoid out-of-bounds should be added
    # The image is evenly divided into 5 segments vertically, from top to bottom: seg 1 2 3 4 5
    # If the center of the closest pole lies inside seg 1 or 5, go straight
    # If the center of the closest pole lies inside seg 2, turn right
    # If the center of the closest pole lies inside seg 4, turn left
    # If the center of the closest pole lies inside seg 3 which is the center of the image, turn left(right)
    heading = 'Go straight'
    if index_max_area != 10:
        cloest_pole = cnts[index_max_area]
        min_y = min(cloest_pole[:,0,1])
        max_y = max(cloest_pole[:,0,1])
        center = 0.5*(min_y+max_y)
        if 0 < center <= h/5 or 4*h/5 < center < h:
            heading = "Go straight"
        elif h/5 < center <= 2*h/5:
            heading = "Turn right"
        elif 3*h/5 < center <= 4*h/5:
            heading = "Turn left"
        else:
            heading = "Turn left"
    text = str(how_many_poles) + ' poles ' + heading
    cv2.putText(image_undist, text, (0,100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 240, 10))
    cv2.imshow('frame', image_undist)
    cv2.waitKey(100)

