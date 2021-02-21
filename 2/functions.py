# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:38:50 2021

@author: leyuan
"""

import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:  #[batch_size, p]
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y

    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))


def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 将one-hot coding转化成单个coding
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size