# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:16:33 2021

@author: leyuan
"""

import numpy as np
from two_layer_net import TwoLayerNet

# 生成数据
x = np.random.randn(3, 784)
t = np.random.randint(0, 10, (3, ))

# 创建网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

grad_numerical = network.numerical_gradient(x, t)
grad_backprop = network.gradient(x, t)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))


# break symmetry
# x = np.random.randn(3, 2)
# t = np.random.randint(0, 10, (3, ))

# # 创建网络
# network = TwoLayerNet(input_size=2, hidden_size=3, output_size=10)
# grad = network.gradient(x, t)
# for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= 0.1 * grad[key]

# for key in grad.keys():
    
#     print(key + ':' + str(grad[key]))