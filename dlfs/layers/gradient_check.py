#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

epsilon = 1e-9
tolerance = 1e-5


def check_parameter_gradient(layer, index, bottom, top_grad):
    top = layer.forward(bottom)
    layer.backward(top_grad)
    p = layer.param[index]
    p_backup = p.copy()
    p_grad = layer.param_grad[index]
    p_grad_ = np.zeros(p.shape)
    for j in range(p.size):
        p[...] = p_backup
        p.reshape(-1)[j] += epsilon
        top_ = layer.forward(bottom)
        layer.backward(top_grad)
        p_grad_.reshape(-1)[j] = ((top_[0] * top_grad[0]).sum() - (top[0] * top_grad[0]).sum()) / epsilon
    p[...] = p_backup
    diff = np.abs(p_grad_ - p_grad).mean() / np.abs(p_grad).mean()
    return diff


def check_bottom_gradient(layer, index, bottom, top_grad):
    top = layer.forward(bottom)
    layer.backward(top_grad)
    x = layer.bottom[index]
    x_backup = x.copy()
    x_grad = layer.bottom_grad[index]
    x_grad_ = np.zeros(x.shape)
    for j in range(x.size):
        x[...] = x_backup
        x.reshape(-1)[j] += epsilon
        top_ = layer.forward(bottom)
        layer.backward(top_grad)
        x_grad_.reshape(-1)[j] = ((top_[0] * top_grad[0]).sum() - (top[0] * top_grad[0]).sum()) / epsilon
    x[...] = x_backup
    diff = np.abs(x_grad_ - x_grad).mean() / np.abs(x_grad).mean()
    return diff
