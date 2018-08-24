import numpy as np
import tensorflow as tf

class TrafficSignClassifierUtil():
    
    @staticmethod
    def get_num_classes(dataset):
        return np.unique(dataset['labels']).shape[0]
    
    @staticmethod
    def get_weights(shape, mu=0, sigma=0.1):
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))

    @staticmethod
    def get_bias(shape, start_val=0.1):
        return tf.Variable(tf.constant(start_val, shape=shape))

    @staticmethod
    def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
        return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')