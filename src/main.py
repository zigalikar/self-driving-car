import os
import numpy as np

from tqdm import tqdm
from random import shuffle

# Loading config
from other.config import Config

# Loading modules
from modules.traffic_sign_classifier.traffic_sign_classifier import TrafficSignClassifier

config = Config('config.json')
tfc = TrafficSignClassifier('traffic_sign_classifier', config)