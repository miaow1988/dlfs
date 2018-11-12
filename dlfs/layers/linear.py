#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Linear():
    def __init__(self, num_input, num_output):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        weight = np.random.randn(num_output, num_input) / num_input * 2
        bias = np.zeros(num_output)
        self.param = [weight, bias]
        self.param_grad = [np.zeros(p.shape) for p in self.param]

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        x = self.bottom[0]
        assert len(x.shape) == 2
        weight, bias = self.param

        y = x.dot(weight.T) + bias
        self.top = [y]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 1
        y_grad = self.top_grad[0]
        assert len(y_grad.shape) == 2
        x = self.bottom[0]
        assert len(x.shape) == 2
        weight, bias = self.param
        weight_grad, bias_grad = self.param_grad

        weight_grad[...] = y_grad.T.dot(x)
        bias_grad[...] = y_grad.sum(0)
        x_grad = y_grad.dot(weight)
        self.bottom_grad = [x_grad]
        return self.bottom_grad
