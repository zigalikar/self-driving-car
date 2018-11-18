import cv2
import numpy as np

import matplotlib.pyplot as plt

import lane_finder.lane_finder_helper as helper

def start():
    ## Processing an image
    # helper.process_image('data/lane-finder/lane-finder_test_image.jpg')
    # helper.process_image('data/lane_recognizer/test/IMG_20180906_063138.jpg')

    ## Processing a video
    helper.process_video('data/lane_finder/lane_finder_test_video.mp4')
