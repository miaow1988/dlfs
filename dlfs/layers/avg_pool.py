#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .img2cols import img2cols, reverse_img2cols


class AvgPool():
    def __init__(self, kernel_size, stride, padding):
        self.bottom = None
        self.bottom_grad = None
        self.propagation = [True]

        self.top = None
        self.top_grad = None

        self.param = []
        self.param_grad = [np.zeros(p.shape) for p in self.param]

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, bottom):
        self.bottom = bottom
        assert len(self.bottom) == 1
        x = self.bottom[0]
        assert x.ndim == 4

        self.batch_size, self.num_input, self.input_h, self.input_w = x.shape
        self.output_h = (self.input_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_w = (self.input_w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.x_shape = list(x.shape)
        self.x_shape[2] += 2 * self.padding
        self.x_shape[3] += 2 * self.padding
        self.x_blocks = img2cols(x, self.kernel_size, self.stride, self.padding)
        y = self.x_blocks.mean(axis=(4, 5))
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

        k = self.kernel_size * self.kernel_size
        x_grad_blocks = y_grad.transpose((2, 3, 0, 1))
        x_grad_blocks = x_grad_blocks[:, :, :, :, None].repeat(k, axis=4) / k
        x_grad_blocks = x_grad_blocks.reshape(
            self.batch_size, self.output_h, self.output_w,
            self.num_input, self.kernel_size, self.kernel_size)
        x_grad = reverse_img2cols(
            self.x_shape, x_grad_blocks,
            self.kernel_size, self.stride, self.padding)
        self.bottom_grad = [x_grad]
        return self.bottom_grad
