import os

from PIL import Image
import numpy as np

class TfcLoader():

    def __init__(self):
        pass
    
    # The main function that loads the dataset
    def load(self, path):
        data = np.array([])

        for info in os.walk(path):
            data = np.append(data, self.read_files_in_dir(info[0]))

        return data
    
    # Reads all files in dir
    def read_files_in_dir(self, dir):
        data = np.array([])

        for name in os.listdir(dir):
            _, ext = os.path.splitext(name)
            
            if ext == '.ppm':
                path = dir + '\\' + name
                data = np.append(data, self.preprocess(Image.open(path)))
        
        return data
    
    ## TODO: preprocessing
    def preprocess(self, data):
        return data