#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Sigmoid():
    def __init__(self):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        self.param = []
        self.param_grad = [np.zeros(p.shape) for p in self.param]

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        self.top = [1.0 / (1 + np.exp(-self.bottom[0]))]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 1
        grad_y = self.top_grad[0]
        y = self.top[0]
        self.bottom_grad = [grad_y * y * (1 - y)]
        return self.bottom_grad
