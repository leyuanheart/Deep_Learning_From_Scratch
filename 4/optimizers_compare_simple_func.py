# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:33:52 2021

@author: leyuan
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizers import SGD, Momentum, AdaGrad, Adam


def func(x,y):
    return x**2/20 + y**2

def g_f(x, y):
    return np.array([x/10, 2*y])

x = np.arange(-10, 11., 0.5)
y = np.arange(-5, 6., 0.5)

X, Y = np.meshgrid(x, y)
Z = func(X, Y)


# 等高线图
plt.figure()
c = plt.contour(X, Y, Z, levels=50)

# 三维图
fig = plt.figure()
axe = plt.axes(projection='3d')
axe.contour3D(X, Y, Z, levels=100)


# 梯度图
grads = g_f(X.flatten(), Y.flatten())
plt.figure()
plt.quiver(X.flatten(), Y.flatten(), -grads[0], -grads[1],  angles="uv",color="#666666")



# ============================================================================================
init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
lr_list = [0.95, 0.1, 1.5, 0.3]
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = g_f(params['x'], params['y'])
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = func(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key + '  lr: '+ str(lr_list[idx-2]))
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()



















