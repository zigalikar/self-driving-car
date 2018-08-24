import os
import numpy as np

from tqdm import tqdm
from random import shuffle

# Loading config
from other.config import Config

# Loading modules
from modules.traffic_sign_classifier.traffic_sign_classifier import TrafficSignClassifier

# Loading loaders
from loaders.tfc_loader import TfcLoader

# Initializing config and modules
config = Config('config.json')
tfc = TrafficSignClassifier('traffic_sign_classifier', TfcLoader(), config)
tfc.init_training()