# fisher.py  fisher线性判别分析

import numpy as np
import matplotlib.pyplot as plt


# 对于watermelon3.txt的解释
# 每行有三个值，前两个值为密度、含糖率，第三个值为标签，1表示好瓜，0表示坏瓜

# 读入数据集并按标签分类
def loaddataset():
    datamat0 = []  # 数据集0 坏瓜
    datamat1 = []  # 数据集1 好瓜
    f = open('watermelon3.txt')  # 读入txt文件
    for line in f.readlines():
        linearr = line.strip().split()  # 每一行按空格分割
        if int(linearr[2]) == 0:
            datamat0.append((np.array([float(linearr[0]), float(linearr[1])])).reshape(2, 1))  # 加入数据集 注意转换为列向量
        else:
            datamat1.append((np.array([float(linearr[0]), float(linearr[1])])).reshape(2, 1))
    return np.array(datamat0), np.array(datamat1)  # 转换为numpy矩阵


# 类均值向量
def avg_vector(mat0, mat1):
    m0 = np.array([[0], [0]])
    n0 = np.shape(mat0)[0]  # 项数
    for i in range(0, n0):
        m0 = m0 + mat0[i]
    m0 = m0 / n0
    m1 = np.array([[0], [0]])
    n1 = np.shape(mat1)[0]
    for i in range(0, n1):
        m1 = m1 + mat1[i]
    m1 = m1 / n1
    return m0, m1


# 类内离散度矩阵
def in_matrix(mat0, mat1, m0, m1):
    s0 = np.zeros((2, 2))
    n0 = np.shape(mat0)[0]
    for i in range(0, n0):
        s0 = s0 + np.dot(mat0[i] - m0, np.transpose(mat0[i] - m0))  # 矩阵乘矩阵的转置
    s1 = np.zeros((2, 2))
    n1 = np.shape(mat1)[0]
    for i in range(0, n1):
        s1 = s1 + np.dot(mat1[i] - m1, np.transpose(mat1[i] - m1))
    sw = s0 + s1
    return sw


# 类间离散度矩阵
def out_matrix(mat0, mat1):
    sb = np.dot(mat0 - mat1, np.transpose(mat0 - mat1))
    return sb


# 方向向量
def direction_vector(sw, m0, m1):
    w = np.dot(np.linalg.inv(sw), m0 - m1)  #矩阵的逆
    return w


# 投影后均值  用于判别
def avg_after_shadow(w, m0, m1):
    m00 = np.dot(np.transpose(w), m0)
    m11 = np.dot(np.transpose(w), m1)
    w0 = -(m00 + m11) / 2
    return w0


# 可视化
def display(mat0, mat1, w):
    fig = plt.figure()
    type1 = []
    type2 = []
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    n0 = np.shape(mat0)[0]
    for i in range(0, n0):
        type2 = plt.scatter(data0[i, 0], data0[i, 1], c='r')
    n1 = np.shape(mat1)[0]
    for i in range(0, n1):
        type1 = plt.scatter(mat1[i, 0], mat1[i, 1], c='b')
    x = np.arange(0, 1, 0.001)
    y = w[1] * x / w[0]
    plt.plot(x, y, c='black')
    plt.legend(handles=[type1, type2], labels=["好瓜", "坏瓜"], loc='best')
    plt.title("线性判别分析")
    plt.show()


if __name__ == '__main__':
    data0, data1 = loaddataset()
    m0, m1 = avg_vector(data0, data1)
    sw = in_matrix(data0, data1, m0, m1)
    sb = out_matrix(m0, m1)
    w = direction_vector(sw, m0, m1)
    print(w)
    display(data0, data1, w)
