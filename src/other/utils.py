import cv2
import numpy as np

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