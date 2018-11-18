import cv2
import numpy as np

import matplotlib.pyplot as plt

def start():
    path = 'data/sign_recog/'

    # Load images
    img1 = cv2.imread(path + 'scene.png', 0) # scene/query image
    img2 = cv2.imread(path + 'train.png', 0) # train image

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None) 

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
    plt.imshow(img3),plt.show()
