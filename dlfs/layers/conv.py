#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .img2cols import img2cols, reverse_img2cols


def view_as_windows(arr_in, window_shape, step=1):
    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    arr_in = np.ascontiguousarray(arr_in)

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = ((
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)) +
                         1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = np.lib.stride_tricks.as_strided(
        arr_in, shape=new_shape, strides=strides)
    return arr_out


class Conv():
    def __init__(self, num_input, num_output, kernel_size, stride, padding):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        weight = np.random.randn(num_output, num_input, kernel_size,
                                 kernel_size)
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

        self.batch_size, _, self.input_h, self.input_w = x.shape
        self.output_h = (self.input_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_w = (self.input_w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.x_shape = list(x.shape)
        self.x_shape[2] += 2 * self.padding
        self.x_shape[3] += 2 * self.padding
        self.x_blocks = img2cols(x, self.kernel_size, self.stride, self.padding)
        self.x_blocks = self.x_blocks.reshape(
            self.output_h, self.output_w, self.batch_size,
            self.num_input * self.kernel_size * self.kernel_size)
        y = self.x_blocks.dot(weight.reshape(self.num_output, -1).T) + bias
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

        self.x_blocks = self.x_blocks.reshape(
            self.output_h * self.output_w * self.batch_size,
            self.num_input * self.kernel_size * self.kernel_size)
        y_grad_blocks = y_grad.transpose((2, 3, 0, 1)).reshape(-1, self.num_output)
        weight_grad[...] = y_grad_blocks.T.dot(self.x_blocks).reshape(weight.shape)
        bias_grad[...] = y_grad_blocks.sum(0)

        x_grad_blocks = y_grad_blocks.dot(weight.reshape(self.num_output, -1))
        x_grad_blocks = x_grad_blocks.reshape(
            self.batch_size, self.output_h, self.output_w,
            self.num_input, self.kernel_size, self.kernel_size)
        x_grad = reverse_img2cols(self.x_shape, x_grad_blocks, self.kernel_size, self.stride, self.padding)
        self.bottom_grad = [x_grad]
        return self.bottom_grad
