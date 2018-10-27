import cv2
import numpy as np

import matplotlib.pyplot as plt

## Preprocesses the image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converting image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # gaussian blur to reduce noise
    return cv2.Canny(blur, 50, 150) # apply the canny function for edge detection

## Applies the lines to the image and shows the modified image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # draw a line segment on the black image we created

    return line_image

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
canny = preprocess(lane_image) # applying the filters to the image
cropped_image = region_of_interest(canny) # crop the image
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) # apply the hough transform
line_image = display_lines(lane_image, lines) # display lines on the black lane image
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # combine the original lane image with the lines image

## Displaying the image
cv2.imshow('result', combo_image) # show the test image
cv2.waitKey(0) # wait until a key is pressed to hide the window
