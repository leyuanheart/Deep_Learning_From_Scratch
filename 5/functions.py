# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:38:50 2021

@author: leyuan
"""

import numpy as np


def smooth_curve(x):
    """
    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """

    Parameters
    ----------
    x : 数据
    t : 标签

    Returns
    -------
    x, t : 
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t



def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


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
    


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由（数据量，通道，高，长）的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    '''
    上面那段构造col的代码不是很好懂，也可能是我自己的逻辑没理清，我按照自己的逻辑写了下面代码，
    结果已经验证过是一样的了，供参考
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(out_h):
        y_start =  stride * y
        y_end = y_start + filter_h
        for x in range(out_w):
            x_start = stride * x
            x_end = x_start + filter_w
            col[:, :, :, :, y, x] = img[:, :, y_start:y_end, x_start:x_end]
    '''
            
    

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入的形状（例：(10, 1, 28, 28)）
    filter_h
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    '''
     和img2col同样的，我也没能理解上面的逻辑，还是按自己的逻辑写了一个       
    img = np.zeros((N, C, stride * (out_h - 1) + filter_h + 1, stride * (out_w - 1) + filter_w + 1))
    for y in range(out_h):
        y_start =  stride * y
        y_end = y_start + filter_h
        for x in range(out_w):
            x_start = stride * x
            x_end = x_start + filter_w
            img[:, :, y_start:y_end, x_start:x_end] = col[:, :, :, :, y, x]
    '''

    return img[:, :, pad:H + pad, pad:W + pad]

