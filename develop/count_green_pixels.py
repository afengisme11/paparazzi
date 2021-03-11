# -*- coding: utf-8 -*-
"""
Created on Tue Mar 9 16:50:21 2021

@author: cheng liu
"""

import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# TODO: Add threshold (minimum) of the number of green pixels to decide whether to turn left/right

folder_path = "green_test_graphs/"
filenames = os.listdir(folder_path)

print("files for test:", filenames)

# YUV range for green
GREEN_MIN = np.array([50, 0, 0], np.uint8)
GREEN_MAX = np.array([200, 120, 135], np.uint8)

# Count the number of pixels that satisfies the green range
for _file in filenames:
    if _file.split('.')[-1] == 'jpg':
        image = cv2.imread(folder_path+_file)
        print(image.shape)

        # The original size of the image is [520*240], only the middle bottom
        # of the image is relevant. Crop the image such that only 
        # [130:390 in width, 0:200 in height] will be analyzed.
        img = image[130:390, 0:200]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # convert to YUV
        dst = cv2.inRange(img, GREEN_MIN, GREEN_MAX) # output 0 for pixels without green
        num_green = cv2.countNonZero(dst) # count pixels with green

        print("{} has {} pixels with green".format(_file, num_green))

# We can count the number of pixels with green and tell the UAV to change
#direction when cnt is smaller than the threshold.

