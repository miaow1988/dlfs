#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time
import collections

import numpy as np


def init_logging(path=None, level=logging.INFO):
    formatter = logging.Formatter(
        '[%(levelname)s] [%(asctime)s] %(message)s',
        '%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(level)

    if path is not None:
        handler = logging.FileHandler(path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class Meter:

    def __init__(self, size=128):
        self.queue = collections.deque(maxlen=size)
        self.t = 0

    def update(self, value):
        self.queue.append(value)

    def __call__(self):
        return np.array(self.queue).mean().item()

    def tic(self):
        self.t = time.time()

    def toc(self):
        self.update(time.time() - self.t)
