# bp.py

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import math
import matplotlib.pyplot as plt


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
iris = load_iris()
X = iris.data
y = iris.target
data = X[y < 2, :2]  # 只取前两列
label = y[y < 2]  # 只取前两类

# 归一化
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
m, n = np.shape(data)

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

# # bp算法
# while maxIter > 0:
#     maxIter -= 1
#     sumE = 0
#     for i in range(m):
#         alpha = np.dot(data[i], v)  # shape=1*q
#         b = sigmoid(alpha - gamma, 1)  # b=f(alpha-gamma), shape=1*q
#         beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
#         predicted = sigmoid(beta - theta, 1)  # shape=1*l ,5.3式
#         E = sum((predicted - label[i]) * (predicted - label[i])) / 2  # 5.4
#         sumE += E  # 5.16
#         # p104
#         g = predicted * (1 - predicted) * (label[i] - predicted)  # shape=1*l 5.10
#         e = b * (1 - b) * (np.dot(w, g.T)).T  # shape=1*q , 5.15
#         w += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # 5.11
#         theta -= eta * g  # 5.12
#         v += eta * np.dot(data[i].reshape((d, 1)), e.reshape((1, q)))  # 5.13
#         gamma -= eta * e  # 5.14


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


ans = predict(data)
anss = []

for value in ans:
    if value >= 0.5:
        anss.append(1)
    else:
        anss.append(0)

for i in range(len(anss)):
    if anss[i]==0:
        plt.scatter(data[i][0], data[i][1], color="red")  # 所有0类的点，用红色
    else:
        plt.scatter(data[i][0], data[i][1], color="blue")  # 所有0类的点，用红色

plt.show()

