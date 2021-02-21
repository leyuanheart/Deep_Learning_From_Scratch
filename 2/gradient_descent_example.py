# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:48:12 2021

@author: leyuan
"""

import numpy as np
import matplotlib.pyplot as plt


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


def f(x):
    '''
    f(x1, x2) = x1^2 + x2^2
    '''
    return x[0]**2 + x[1]**2



def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


if __name__ == '__main__':

    init_x = np.array([-3.0, 4.0])    
    
    lr = 0.1
    step_num = 20
    x, x_history = gradient_descent(f, init_x, lr=lr, step_num=step_num)
    
    x0 = np.arange(-4, 4, 0.25)
    x1 = np.arange(-4, 4, 0.25)
    X0, X1 = np.meshgrid(x0, x1)
    Y = f(np.array([X0,X1]))
    
    plt.figure(figsize=(8, 8))
    c = plt.contour(X0, X1, Y, levels=[5, 10, 15], linestyles='--')
    plt.clabel(c, fontsize=10, colors='k', fmt='%.1f')
    # plt.plot( [-5, 5], [0,0], '--b')
    # plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')
    
    # plt.xlim(-6, 6)
    # plt.ylim(-6, 6)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()