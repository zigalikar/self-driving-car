import json
import os.path

class Config(object):
    
    def __init__(self, file):
        if os.path.isfile(file):
            self.__dict__ = json.loads(open(file, encoding='utf-8', errors='ignore').read())
            print('Loaded config from file: \'' + file + '\'.')
        else:
            print('Error loading config from file: \'' + file + '\'.')