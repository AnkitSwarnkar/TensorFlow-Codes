# -*- coding: utf-8 -*-
"""
@author: ankitswarnkar
"""
import numpy as np
import tensorflow as tf
# Load data
from tensorflow.examples.tutorials.mnist import input_data

class load_data:
    def __init__(self):
        self.mnist = input_data.read_data_sets("data/", one_hot=True)
    def return_data(self):
            X_train = self.mnist.train.images
            Y_train = self.mnist.train.labels
            X_test = self.mnist.test.images
            Y_test = self.mnist.test.labels
            return X_train, Y_train, X_test,Y_test
