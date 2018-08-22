import os

from PIL import Image
import numpy as np

from loaders.loader_interface import LoaderInterface

class TfcLoader(LoaderInterface):

    def __init__(self):
        pass
    
    def load(self, path):
        # #subdirs = [x[0] for x in os.walk(data_input)]

        # #print(subdirs)
        # test_dir = data_input + '\\00000.ppm'
        # print(test_dir)
        # if (os.path.isfile(test_dir)):
        #     contents = np.array(Image.open(test_dir))
        #     print(contents)

        print('Load implementation')
        return ['test', 'test2']