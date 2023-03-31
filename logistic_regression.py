# logistic_regression.py 对数线性回归

import numpy as np
import matplotlib.pyplot as plt


# 对于watermelon3.txt的解释
# 每行有三个值，前两个值为密度、含糖率，第三个值为标签，1表示好瓜，0表示坏瓜

# 读入数据集
def loaddataset():
    datamat = []  # 数据集
    labelmat = []  # 标签集
    f = open('watermelon3.txt')  # 读入txt文件
    for line in f.readlines():
        linearr = line.strip().split()  # 每一行按空格分割
        datamat.append([1.0, float(linearr[0]), float(linearr[1])])  # 加入数据集 x0设为1.0 方便计算
        labelmat.append(int(linearr[2]))  # 加入标签
    return datamat, labelmat


# sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 回归梯度上升算法
def gradascent(datamat_in, labelmat_in):
    # 转换为numpy矩阵
    datamatrix = np.mat(datamat_in)
    labelmartix = np.mat(labelmat_in).transpose()
    m, n = np.shape(datamatrix)
    alpha = 0.001  # 移动步长
    maxcycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 权重
    for k in range(maxcycles):
        # 矩阵相乘
        h = sigmoid(datamatrix * weights)
        error = (labelmartix - h)
        weights = weights + alpha * datamatrix.transpose() * error
    return weights


# 绘图
def display(weights):
    datamat, labelmat = loaddataset()
    dataarr = np.array(datamat)  # 列表转数组
    n = np.shape(datamat)[0]
    # 将两类划分
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xcord1.append(dataarr[i, 1])
            ycord1.append(dataarr[i, 2])
        else:
            xcord2.append(dataarr[i, 1])
            ycord2.append(dataarr[i, 2])
    fig = plt.figure()
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    # 散点图
    type1 = plt.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    type2 = plt.scatter(xcord2, ycord2, s=30, c='green')
    # 最佳拟合直线
    x = np.arange(0.0, 1.0, 0.001)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)
    plt.legend(handles=[type1, type2], labels=["好瓜", "坏瓜"], loc='best')
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.title("对数线性回归")
    plt.show()


# 主函数
def main():
    data, label = loaddataset()  # 读入数据集与标签集
    print(data)
    print(label)
    weight = (gradascent(data, label))  # 梯度上升算法
    display(weight.getA())  # 绘图


if __name__ == '__main__':
    main()
