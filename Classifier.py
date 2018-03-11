# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 02:36:28 2018

@author: Usman Ashraf
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def sigmoid(var):
  return 1 / (1 + np.exp(-var))


def Function(weight, var, num):
    res = np.dot(weight, var.T) + num
    return sigmoid(res)


def Calculate_Cost(weight, var, yvar, bvar):
    hyp = Function(weight, var, bvar)
    p1 = np.dot( yvar, np.log(hyp) )
    p2 = np.dot( np.add(1, np.multiply(-1, yvar)), np.log(np.add(1, np.multiply(-1, hyp))) )
    Z = np.sum(np.add(p1, p2))
    return np.multiply(-1/var.shape[0] , Z)
    

Column_names = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','Result']
df = pd.read_csv("data.csv", names = Column_names)

random.seed(1)
np.random.seed(1)

weights = np.array([np.random.rand(9)])
x_values = df[['L1','L2','L3','L4','L5','L6','L7','L8','L9']].as_matrix()
y_values = df[['Result']].as_matrix()
temp = weights.copy()
alpha = [0.001, 0.0001, 0.00001, 0.0000001] # As we decrease the alpha loss rate decreases slowly 
lossHistory = [[], [], [], []]

sett = 0

for i in range(len(alpha)):
    weight = temp.copy()
    for iteration in range(100):    # increasing number of iterations decreases loss rate quickly.
        Shap = x_values.shape[0]
        ret = Function(weight, x_values, sett)
        dat = ret - y_values
        w = 0
        w = np.sum(np.dot(x_values.T, dat))
        w /= Shap
        b = 0
        b = np.sum(dat)
        b /= Shap
        weight =  weight - alpha[i] * w
        sett =  sett - alpha[i] * b
        lossHistory[i].append(Calculate_Cost(weight, x_values, y_values, sett))

    plt.plot(lossHistory[i])


plt.ylabel('Loss Rate')
plt.xlabel('Alpha Index')       # Ploting different values of alpha
plt.legend(('0.001', '0.0001', '0.00001', '0.000001'))   
plt.show()