# bp.py

import pandas as pd
import numpy as np
import random
import math


# sigmoid函数
def sigmoid(iX, dimension):  # 带维度矩阵
    if dimension == 1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i], dimension - 1)
    return iX


# 读入数据
dataset = pd.read_csv('watermelon_3.csv', delimiter=",")
# 创建文字与数值对应字典
attributeMap = {'浅白': 0, '青绿': 0.5, '乌黑': 1, '蜷缩': 0, '稍蜷': 0.5, '硬挺': 1, '沉闷': 0, '浊响': 0.5, '清脆': 1, '模糊': 0,
                '稍糊': 0.5, '清晰': 1, '凹陷': 0, '稍凹': 0.5, '平坦': 1, '硬滑': 0, '软粘': 1, '否': 0, '是': 1}
del dataset['编号']
dataset = np.array(dataset)
m, n = np.shape(dataset)
# 根据文字转换成数据
for i in range(m):
    for j in range(n):
        if dataset[i, j] in attributeMap:
            dataset[i, j] = attributeMap[dataset[i, j]]
        dataset[i, j] = round(dataset[i, j], 3)
# 分割为标签集与数据集
label = dataset[:, n - 1]
data = dataset[:, :n - 1]
m, n = np.shape(data)
print(label)
print(data)

# 参数设置
d = n  # 输入维度
l = 1  # 输出维度
q = d + 1  # 隐层结点数量
theta = [random.random() for i in range(l)]  # the threshold of the output nodes
gamma = [random.random() for i in range(q)]  # 隐层阈值
# 随机权值
v = [[random.random() for i in range(q)] for j in range(d)]
w = [[random.random() for i in range(l)] for j in range(q)]
eta = 0.2  # 学习率
maxIter = 10000  # 迭代次数

# bp算法
while maxIter > 0:
    maxIter -= 1
    sumE = 0
    for i in range(m):
        alpha = np.dot(data[i], v)  # shape=1*q
        b = sigmoid(alpha - gamma, 1)  # b=f(alpha-gamma), shape=1*q
        beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
        predicted = sigmoid(beta - theta, 1)  # shape=1*l ,5.3式
        E = sum((predicted - label[i]) * (predicted - label[i])) / 2  # 5.4
        sumE += E  # 5.16
        # p104
        g = predicted * (1 - predicted) * (label[i] - predicted)  # shape=1*l 5.10
        e = b * (1 - b) * (np.dot(w, g.T)).T  # shape=1*q , 5.15
        w += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # 5.11
        theta -= eta * g  # 5.12
        v += eta * np.dot(data[i].reshape((d, 1)), e.reshape((1, q)))  # 5.13
        gamma -= eta * e  # 5.14


# 累积bp算法
label = label.reshape((m, l))
while maxIter > 0:
    maxIter -= 1
    sumE = 0
    alpha = np.dot(data, v)  # shape=m*q
    b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predicted = sigmoid(beta - theta, 2)  # shape=m*l ,5.3
    E = sum(sum((predicted - label) * (predicted - label))) / 2  # 5.4
    g = predicted * (1 - predicted) * (label - predicted)  # shape=m*l .10
    e = b * (1 - b) * (np.dot(w, g.T)).T  # shape=m*q , p104--5.15
    w += eta * np.dot(b.T, g)  # 5.11 shape (q*l)=(q*m) * (m*l)
    theta -= eta * g  # 5.12
    v += eta * np.dot(data.T, e)  # 5.13 (d,q)=(d,m)*(m,q)
    gamma -= eta * e  # 5.14


# 预测函数
def predict(iX):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3
    return predictY


print(predict(data))
