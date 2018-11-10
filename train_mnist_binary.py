#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import dlfs
import dlfs.layers as L


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    dlfs.utils.init_logging(level=logging.INFO)

    mnist = dlfs.DatasetMNIST(args.batch_size, 'data/mnist', is_binary=True)

    meter_loss = dlfs.utils.Meter()
    meter_acc = dlfs.utils.Meter()

    net = (
        L.Linear(28*28, 256),
        L.Sigmoid(),
        L.Linear(256, 128),
        L.Sigmoid(),
        L.Linear(128, 1),
        L.Sigmoid(),
    )
    net_loss = L.CrossEntropyLoss()

    # train
    for global_iter, (data, label) in enumerate(mnist):
        label = label.reshape(-1, 1)

        # forward
        logging.debug('forward')
        x = [data]
        for layer in net:
            x = layer.forward(x)
            logging.debug(x[0].shape)
        loss = net_loss.forward((x[0], label))
        meter_loss.update(loss[0])
        acc = ((x[0] > 0.5) == label).mean()
        meter_acc.update(acc)
        if global_iter % 100 == 0:
            logging.info('iter=%d loss=%.3f acc=%.3f' %
                         (global_iter, meter_loss(), meter_acc()))

        # backward
        logging.debug('backward')
        grad_x = [net_loss.backward([])[0]]
        for layer in net[::-1]:
            grad_x = layer.backward(grad_x)
            logging.debug(grad_x[0].shape)

        # update
        lr = 0.01
        for layer in net:
            for i, p in enumerate(layer.param):
                p[...] = p - lr * layer.param_grad[i]
