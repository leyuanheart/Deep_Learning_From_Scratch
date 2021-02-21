# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:29:37 2021

@author: leyuan
"""

import os
import sys

sys.path.append(os.pardir)  
import numpy as np
import matplotlib.pyplot as plt
from data.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from optimizers import SGD
from functions import shuffle_dataset



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了提到训练速度，减少数据
x_train = x_train[:500]
t_train = t_train[:500]

# 划分验证集
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


train_size = x_train.shape[0]
mini_batch_size = 100
epochs = 20


def __train(lr, weight_decay, epocs=50, verbose=False):    # 减少epoch的数量
               
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    
    optimizer = SGD(lr)
    
    iter_per_epoch = max(train_size / mini_batch_size, 1)
    current_iter = 0
    current_epoch = 0
    
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    
    for i in range(int(epochs * iter_per_epoch)):
        batch_mask = np.random.choice(train_size, mini_batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if verbose:
            print("train loss:" + str(loss))
            
        if current_iter % iter_per_epoch == 0:
            current_epoch += 1
            
                
            train_acc = network.accuracy(x_train, t_train)
            val_acc = network.accuracy(x_val, t_val)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            if verbose: 
                print("=== epoch:" + str(current_epoch) + ", train acc:" + str(train_acc) + ", validation acc:" + str(val_acc) + " ===")
        current_iter += 1

    return val_acc_list, train_acc_list


# 超参数探索======================================
'''
针对weight_decay和learning rate
'''
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 超参数范围指定===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 可视化========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()