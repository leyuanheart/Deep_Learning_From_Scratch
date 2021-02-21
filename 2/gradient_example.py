# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:42:06 2021

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


def func(x):
    '''
    f(x) = x^2, 可以适用于1维或者多维的输入
    '''
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
    



if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(func, np.array([X, Y]).T).T
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()