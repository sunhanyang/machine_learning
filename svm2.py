# svm.py 支持向量机

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline


def SVC_(kernel="rbf", gamma=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("linearSVC", SVC(kernel="rbf", gamma=gamma))
    ])


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),  # 600个，影响列数
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),  # 600个，影响行数
    )
    # x0 和 x1 被拉成一列，然后拼接成360000行2列的矩阵，表示所有点
    X_new = np.c_[x0.ravel(), x1.ravel()]  # 变成 600 * 600行， 2列的矩阵

    y_predict = model.predict(X_new)  # 二维点集才可以用来预测
    zz = y_predict.reshape(x0.shape)  # (600, 600)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)


if __name__ == '__main__':
    # 读入数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = X[y < 2, :2]  # 只取前两列
    y = y[y < 2]  # 只取前两类

    # 归一化
    scaler = StandardScaler()
    scaler.fit(X)
    X_stand = scaler.transform(X)
    svc = SVC_(kernel="rbf", gamma=1)
    svc.fit(X_stand, y)

    plot_decision_boundary(svc, [-3, 3, -3, 3])
    plt.scatter(X_stand[y == 0, 0], X_stand[y == 0, 1])
    plt.scatter(X_stand[y == 1, 0], X_stand[y == 1, 1])

    plt.show()
