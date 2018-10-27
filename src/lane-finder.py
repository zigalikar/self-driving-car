import cv2
import numpy as np

import matplotlib.pyplot as plt

import other.lane_finder_helper as helper

## Processing an image
# helper.process_image('data/lane-finder/lane-finder_test_image.jpg')
# helper.process_image('data/lane_recognizer/test/IMG_20180906_063138.jpg')

## Processing a video
helper.process_video('data/lane-finder/lane-finder_test_video.mp4')
