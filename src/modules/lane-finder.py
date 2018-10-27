import cv2
import numpy as np

import matplotlib.pyplot as plt

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # gaussian blur to reduce noise
    return cv2.Canny(blur, 50, 150) # apply the canny function for edge detection

## Loading the test image
image = cv2.imread('data/lane-finder/lane-finder_test_image.jpg') # loading the image from file
lane_image = np.copy(image) # copy so changes on the image we edit do not change the original image
lane_image = preprocess(lane_image) # applying the filters to the image

## Displaying the image
plt.imshow(lane_image) # show the test image
plt.show() # wait until a key is pressed to hide the window

