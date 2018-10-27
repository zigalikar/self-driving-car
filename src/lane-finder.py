import cv2
import numpy as np

import matplotlib.pyplot as plt

import other.utils as utils

## Loading the test image
image = cv2.imread('data/lane-finder/lane-finder_test_image.jpg') # loading the image from file
lane_image = np.copy(image) # copy so changes on the image we edit do not change the original image

## Processing the test image
canny = utils.preprocess(lane_image) # applying the filters to the image
cropped_image = utils.region_of_interest(canny) # crop the image
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) # apply the hough transform
average_lines = utils.average_slope_intercept(lane_image, lines)
line_image = utils.display_lines(lane_image, average_lines) # display lines on the black lane image
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # combine the original lane image with the lines image

## Displaying the image
cv2.imshow('result', combo_image) # show the test image
cv2.waitKey(0) # wait until a key is pressed to hide the window
