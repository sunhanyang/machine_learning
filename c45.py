from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_stand, y)
    plt.scatter(X_stand[y == 0, 0], X_stand[y == 0, 1])
    plt.scatter(X_stand[y == 1, 0], X_stand[y == 1, 1])
    plt.show()
