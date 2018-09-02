import os
import numpy as np

from tqdm import tqdm
from random import shuffle

# Loading config
from other.config import Config

# Loading modules
from modules.traffic_sign_classifier.traffic_sign_classifier import TrafficSignClassifier
from modules.lane_recognizer.lane_recognizer import LaneRecognizer

# Loaders
from loaders.tfc_loader import TfcLoader
from loaders.lr_loader import LrLoader

# Initializing config and modules
config = Config('config.json')
tfc = TrafficSignClassifier('traffic_sign_classifier', TfcLoader(), config)
lrecog = LaneRecognizer('lane_recognizer', LrLoader(), config)