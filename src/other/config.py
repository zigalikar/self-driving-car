import json
import os.path

from other.util import log

class Config(object):
    
    def __init__(self, file):
        if os.path.isfile(file):
            self.__dict__ = json.loads(open(file, encoding='utf-8', errors='ignore').read())
            log('Loaded config from file: \'' + file + '\'.')
        else:
            log('Error loading config from file: \'' + file + '\'.')