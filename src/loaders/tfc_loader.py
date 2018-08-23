import os

import cv2

from PIL import Image
import numpy as np

class TfcLoader():

    def __init__(self):
        pass
    
    def load(self, path):
        raise NotImplementedError()
    
    # Preprocesses the data before entering the neural network model
    def preprocess(self, fts):
        fts = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in fts])
        fts = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in fts])
        fts = np.float32(fts)
        fts -= np.mean(fts, axis=0)
        fts /= (np.std(fts, axis=0) + np.finfo('float32').eps)

        return fts