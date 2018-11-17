#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import numpy as np


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


def img2col(img, kernel_size, stride, padding):
    img = np.pad(
        img, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant')
    col = view_as_windows(
        img, (img.shape[0], img.shape[1], kernel_size, kernel_size),
        (1, 1, stride, stride))
    col = col.reshape(col.shape[2], col.shape[3], col.shape[4], -1)
    return col
