#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class BatchNorm():
    def __init__(self, num_channel):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        weight = np.ones(num_channel)
        bias = np.zeros(num_channel)
        self.param = [weight, bias]
        self.param_grad = None

        mean = np.zeros(num_channel)
        var = np.random.rand(num_channel)
        self.aux_param = [mean, var]

        self.num_channel = num_channel
        self.epsilon = 1e-6
        self.is_train = True

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        x = self.bottom[0]
        assert x.shape[1] == self.num_channel
        weight, bias = self.param
        mean, var = self.aux_param

        x_shape = x.shape
        shape = list(x_shape) + [1] * (4 - len(x_shape))
        x = x.reshape(shape)
        if self.is_train:
            mean[...] = 0.999 * mean + 0.001 * x.mean(axis=(0, 2, 3))
            var[...] = 0.999 * var + 0.001 * x.var(axis=(0, 2, 3))
        x = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var + self.epsilon).reshape(1, -1, 1, 1)
        x = x * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
        x = x.reshape(x_shape)
        self.top = [x]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 1
        y_grad = self.top_grad[0]
        assert len(y_grad.shape) == 2
        x = self.bottom[0]
        assert x.shape[1] == self.num_channel
        weight, bias = self.param
        mean, var = self.aux_param

        x_shape = x.shape
        shape = list(x_shape) + [1] * (4 - len(x_shape))
        x = x.reshape(shape)
        x = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var + self.epsilon).reshape(1, -1, 1, 1)
        y_grad = y_grad.reshape(shape)
        weight_grad = (y_grad * x).sum(axis=(0, 2, 3))
        bias_grad = y_grad.sum(axis=(0, 2, 3))
        self.param_grad = [weight_grad, bias_grad]

        x_grad = y_grad * weight.reshape(1, -1, 1, 1)
        x_grad = x_grad / np.sqrt(var + self.epsilon).reshape(1, -1, 1, 1)

        x = x.reshape(x_shape)
        y_grad = y_grad.reshape(x_shape)
        x_grad = x_grad.reshape(x_shape)
        self.bottom_grad = [x_grad]
        return self.bottom_grad
