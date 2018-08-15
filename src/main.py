import os
import json
import numpy as np

from tqdm import tqdm
from random import shuffle

config = json.loads(open('config.json', encoding='utf-8', errors='ignore').read())