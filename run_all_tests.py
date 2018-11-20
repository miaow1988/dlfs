#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover(
        './test', pattern='test_*.py', top_level_dir=None)
    runner.run(suite)
