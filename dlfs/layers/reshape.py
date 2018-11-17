#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Reshape():
    def __init__(self, top_shape):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        self.param = []
        self.param_grad = [np.zeros(p.shape) for p in self.param]

        self.top_shape = top_shape

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        self.bottom_shape = self.bottom[0].shape
        self.top = [self.bottom[0].reshape(self.top_shape)]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 1
        self.bottom_grad = [self.top_grad[0].reshape(self.bottom_shape)]
        return self.bottom_grad
