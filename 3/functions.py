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



def numerical_gradient(f, x):
    '''
    NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。
    https://blog.csdn.net/m0_37393514/article/details/79563776
    '''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val 
        it.iternext()   
        
    return grad


# def _numerical_gradient_no_batch(f, x):
#     h = 1e-4  # 0.0001
#     grad = np.zeros_like(x)
    
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + h
#         fxh1 = f(x)  # f(x+h)
        
#         x[idx] = tmp_val - h 
#         fxh2 = f(x)  # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         x[idx] = tmp_val  # 还原值
        
#     return grad


# def numerical_gradient(f, X):
#     if X.ndim == 1:
#         return _numerical_gradient_no_batch(f, X)
#     else:
#         grad = np.zeros_like(X)
        
#         for idx, x in enumerate(X):
#             grad[idx] = _numerical_gradient_no_batch(f, x)
        
#         return grad