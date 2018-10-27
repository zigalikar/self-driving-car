import cv2
import numpy as np

import matplotlib.pyplot as plt

import other.utils as utils

def detect_lanes(image):
    canny = utils.preprocess(image) # applying the filters to the image
    cropped_image = utils.region_of_interest(canny) # crop the image
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) # apply the hough transform
    average_lines = utils.average_slope_intercept(image, lines)
    line_image = utils.display_lines(image, average_lines) # display lines on the black lane image
    return cv2.addWeighted(image, 0.8, line_image, 1, 1) # combine the original lane image with the lines image

# ## Loading the test image
# image = cv2.imread('data/lane-finder/lane-finder_test_image.jpg') # loading the image from file
# lane_image = np.copy(image) # copy so changes on the image we edit do not change the original image

# ## Processing the test image
# processed_image = detect_lanes(lane_image)

# ## Displaying the image
# cv2.imshow('result', processed_image) # show the test image
# cv2.waitKey(0) # wait until a key is pressed to hide the window

## Process the test video
cap = cv2.VideoCapture('data/lane-finder/lane-finder_test_video.mp4') # opens the test video
while (cap.isOpened()):
    _, frame = cap.read() # reads each frame while the video is open
    processed_frame = detect_lanes(frame)
    cv2.imshow('result', processed_frame) # show the test image
    if cv2.waitKey(1) == ord('a'): # wait until a key is pressed to hide the window
        break

## Dispose of the video capture
cap.release()
cv2.destroyAllWindows()