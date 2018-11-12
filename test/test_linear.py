#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np

import dlfs.layers
from dlfs.layers import check_parameter_gradient
from dlfs.layers import check_bottom_gradient


class TestLinear(unittest.TestCase):
    layer = dlfs.layers.Linear(num_input=3, num_output=4)
    for p in layer.param:
        p[...] = np.random.rand(*p.shape)
    bottom = [np.random.randn(2, 3)]
    top = layer.forward(bottom)
    top_grad = [np.ones(t.shape) for t in top]

    def test_backward_param(self):
        for i in range(len(self.layer.param)):
            diff = check_parameter_gradient(
                self.layer, i, self.bottom, self.top_grad
            )
            self.assertLess(diff, 1e-5)

    def test_backward_bottom(self):
        for i in range(len(self.layer.bottom)):
            if self.layer.propagation[i] is False:
                continue
            diff = check_bottom_gradient(
                self.layer, i, self.bottom, self.top_grad
            )
            self.assertLess(diff, 1e-5)


if __name__ == '__main__':
    unittest.main()
