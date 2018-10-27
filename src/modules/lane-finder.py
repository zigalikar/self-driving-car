import cv2
import numpy as np

import matplotlib.pyplot as plt

## Preprocesses the image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # gaussian blur to reduce noise
    return cv2.Canny(blur, 50, 150) # apply the canny function for edge detection

## Returns the enclosed region of view
def region_of_interest(image):
    height = image.shape[0] # get the height of the image
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ]) # create the region of interest - an array of polygons (1 polygon) because the fillPoly function requires an array
    mask = np.zeros_like(image) # creates an array of zeros with the same shape as the input image - create a BLACK mask
    cv2.fillPoly(mask, polygons, 255) # apply the triangle of color 255 (white) on the mask
    masked_image = cv2.bitwise_and(image, mask) # masking the image with the bitwise function
    return masked_image

## Loading the test image
image = cv2.imread('data/lane-finder/lane-finder_test_image.jpg') # loading the image from file
lane_image = np.copy(image) # copy so changes on the image we edit do not change the original image
lane_image = preprocess(lane_image) # applying the filters to the image

## Displaying the image
cv2.imshow('result', region_of_interest(lane_image)) # show the test image
cv2.waitKey(0) # wait until a key is pressed to hide the window
