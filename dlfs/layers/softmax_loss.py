#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class SoftmaxLoss():
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
        assert len(x.shape) == 2
        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        assert np.all((y >= 0) & (y < x.shape[1]))

        x_exp = np.exp(x)
        x_softmax = x_exp / x_exp.sum(1)[:, None]
        x_softmax = x_softmax[np.arange(x.shape[0]), y]
        loss = -np.log(x_softmax).mean()
        self.top = [loss]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 0
        x, y = self.bottom

        y_onehot = np.arange(x.shape[1])[None, :] == y[:, None]
        x_exp = np.exp(x)
        x_softmax = x_exp / x_exp.sum(1)[:, None]
        x_grad = (x_softmax - y_onehot) / y.size
        self.bottom_grad = [x_grad, None]
        return self.bottom_grad
