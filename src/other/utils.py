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

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters # extract the slope and intercept from the parameters
    y1 = image.shape[0] # height of the image - we want the line to start at the bottom of the image
    y2 = int(y1 * (3/5)) # the lines go 3/5 of the way to the top of the image
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

## Averages left and right lines to get one left and one right line
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # y = m * x + b -> get the slope and the y intercept
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: # left line
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Average for both sides
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])