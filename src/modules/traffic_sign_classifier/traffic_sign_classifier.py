import tensorflow as tf
from tensorflow.contrib.layers import flatten

from modules.module_base import ModuleBase

class TrafficSignClassifier(ModuleBase):

    def __init__(self, module_name, loader, config):
        super().__init__(module_name, loader, config)