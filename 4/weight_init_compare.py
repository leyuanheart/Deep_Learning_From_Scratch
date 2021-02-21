# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:18:02 2021

@author: leyuan
"""

'''
更换不同的权重初始化的值和激活值，可以得到文中的图
'''

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, relu, tanh

input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 权重初始化
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    z = np.dot(x, w)

    # 激活值
    a = sigmoid(z)
    # a = relu(z)
    # a = tanh(z)

    activations[i] = a

# 可视化
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()