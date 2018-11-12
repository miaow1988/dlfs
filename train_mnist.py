#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import numpy as np

import dlfs
import dlfs.layers as L


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    dlfs.utils.init_logging(path='log.txt', level=logging.INFO)

    mnist = dlfs.DatasetMNIST(args.batch_size, 'data/mnist')

    meter_loss = dlfs.utils.Meter()
    meter_acc = dlfs.utils.Meter()

    net = (
        L.Linear(28*28, 256),
        L.BatchNorm(256),
        L.ReLU(),
        L.Linear(256, 256),
        L.BatchNorm(256),
        L.ReLU(),
        L.Linear(256, 128),
        L.BatchNorm(128),
        L.ReLU(),
        L.Linear(128, 10),
    )
    net_loss = L.SoftmaxLoss()

    # parameters
    parameters = []
    parameters_grad = []
    parameters_velocity = []
    for layer in net:
        for i, p in enumerate(layer.param):
            parameters.append(p)
            parameters_grad.append(layer.param_grad[i])
            parameters_velocity.append(np.zeros(p.shape))

    # train
    for global_iter, (data, label) in enumerate(mnist):
        # forward
        logging.debug('forward')
        x = [data]
        for layer in net:
            x = layer.forward(x)
            logging.debug(x[0].shape)
        loss = net_loss.forward((x[0], label))
        meter_loss.update(loss[0])
        prediction = (np.exp(x[0]) / np.exp(x).sum(1)).argmax(1)
        acc = (prediction == label).mean()
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
        lr = 0.001
        wd = 0.0001
        momentum = 0.9
        for i, p in enumerate(parameters):
            p_grad = parameters_grad[i] + wd * p
            p_velocity = parameters_velocity[i]
            p_velocity[...] = momentum * p_velocity + (1 - momentum) * p_grad
            p[...] = p - lr * p_velocity
