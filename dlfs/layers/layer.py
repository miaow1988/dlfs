#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Layer():
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
        self.top = bottom
        return self.top

    def backward(self, top_grad):
        self.top_grad = top_grad
        self.bottom_grad = top_grad
        return self.bottom_grad
