#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import gzip
import argparse

import numpy as np


def load_mnist(data_path='data/mnist'):
    filename = (
        ('train_x', 'train-images-idx3-ubyte.gz'),
        ('test_x', 't10k-images-idx3-ubyte.gz'),
        ('train_y', 'train-labels-idx1-ubyte.gz'),
        ('test_y', 't10k-labels-idx1-ubyte.gz'),
    )
    mnist = {}
    for name in filename[:2]:
        with gzip.open(os.path.join(data_path, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(os.path.join(data_path, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    return mnist


class DatasetMNIST():
    def __init__(self, batch_size, mnist_path, is_binary=False):
        self.batch_size = batch_size
        data_path = os.path.join(mnist_path, 'train-images-idx3-ubyte.gz')
        with gzip.open(data_path, 'rb') as f:
            self.data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
        label_path = os.path.join(mnist_path, 'train-labels-idx1-ubyte.gz')
        with gzip.open(label_path, 'rb') as f:
            self.label = np.frombuffer(f.read(), np.uint8, offset=8)
        assert len(self.data) == len(self.label)

        if is_binary:
            idx = (self.label < 2)
            self.data = self.data[idx]
            self.label = self.label[idx]
        
        # normalization
        self.mean = self.data.mean()
        self.std = (((self.data - self.mean)**2).mean())**0.5
    
    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.choice(len(self.data), size=self.batch_size, replace=False)
        x = self.data[idx]
        x = (x - self.mean) / (self.std + 1e-6)
        x *= 255
        y = self.label[idx]
        return x, y
