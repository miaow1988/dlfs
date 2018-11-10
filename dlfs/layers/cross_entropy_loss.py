#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True, False]

        self.top = None
        self.top_grad = None
        
        self.param = []
        self.param_grad = []
    
    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 2
        x, y = self.bottom
        assert x.shape == y.shape
        assert np.all((x >= 0) & (x <= 1))
        assert np.all((y >= 0) & (y <= 1))

        loss = y * np.log(x) + (1 - y) * np.log(1 - x)
        loss = -loss.mean()
        self.top = [loss]
        return self.top
    
    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 0
        x, y = self.bottom
        assert x.shape == y.shape

        x_grad = y / x - (1 - y) / (1 - x)
        x_grad = -x_grad / x_grad.size

        self.bottom_grad = [x_grad, None]
        return self.bottom_grad
