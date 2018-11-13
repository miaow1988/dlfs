#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numbers
import itertools

import numpy as np


def view_as_windows(arr_in, window_shape, step=1):
    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    arr_in = np.ascontiguousarray(arr_in)

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = np.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


class Conv():
    def __init__(self, num_input, num_output, kernel_size, stride, padding):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        weight = np.random.randn(num_output, num_input, kernel_size, kernel_size)
        bias = np.zeros(num_output)
        self.param = [weight, bias]
        self.param_grad = [np.zeros(p.shape) for p in self.param]

        self.num_input = num_input
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        x = self.bottom[0]
        assert x.ndim == 4
        weight, bias = self.param
        assert weight.shape[1] == x.shape[1]

        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        x_blocks = view_as_windows(x, (x.shape[0], x.shape[1], self.kernel_size, self.kernel_size), (1, 1, self.stride, self.stride))
        x_blocks = x_blocks.reshape(x_blocks.shape[2], x_blocks.shape[3], x_blocks.shape[4], -1)
        y = x_blocks.dot(weight.reshape(self.num_output, -1).T) + bias.reshape(1, 1, 1, self.num_output)
        y = y.transpose((2, 3, 0, 1))
        self.top = [y]
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        assert len(self.top_grad) == 1
        y_grad = self.top_grad[0]
        assert len(y_grad.shape) == 4
        x = self.bottom[0]
        assert len(x.shape) == 4
        weight, bias = self.param
        weight_grad, bias_grad = self.param_grad

        # import ipdb; ipdb.set_trace()
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        batch_size = x.shape[0]
        x_blocks = view_as_windows(x, (batch_size, self.num_input, self.kernel_size, self.kernel_size), (1, 1, self.stride, self.stride))
        output_h, output_w = x_blocks.shape[2], x_blocks.shape[3]
        x_blocks = x_blocks.reshape(output_h, output_w, batch_size, -1)

        x_blocks = x_blocks.reshape(-1, x_blocks.shape[3])
        y_grad_blocks = y_grad.transpose((2, 3, 0, 1)).reshape(-1, self.num_output)
        weight_grad[...] = y_grad_blocks.T.dot(x_blocks).reshape(weight.shape)
        bias_grad[...] = y_grad_blocks.sum(0)

        t = y_grad_blocks.dot(weight.reshape(self.num_output, -1))
        t = t.reshape(batch_size, output_h, output_w, self.num_input, self.kernel_size, self.kernel_size)
        x_grad = np.zeros(x.shape)
        for i, j in itertools.product(range(output_h), range(output_w)):
            i0 = i * self.stride
            i1 = i0 + self.kernel_size
            j0 = j * self.stride
            j1 = j0 + self.kernel_size
            x_grad[:, :, i0:i1, j0:j1] += t[:, i, j, :, :, :]
        if self.padding > 0:
            x_grad = x_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # y_grad = np.pad(y_grad, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # y_grad_blocks = skimage.util.view_as_windows(y_grad, (batch_size, self.num_output, self.kernel_size, self.kernel_size), (1, 1, 1, 1))
        # output_h, output_w = y_grad_blocks.shape[2], y_grad_blocks.shape[3]
        # y_grad_blocks = y_grad_blocks.reshape(output_h, output_w, batch_size, -1)
        # x_grad = y_grad_blocks.dot(np.flip(weight, axis=(2, 3)).transpose((1, 0, 2, 3)).reshape(self.num_input, -1).T)
        # x_grad = x_grad.transpose((2, 3, 0, 1))
        
        self.bottom_grad = [x_grad]
        return self.bottom_grad
