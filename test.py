#导入所需库：
import numpy as np
from matplotlib import pyplot as plt

#创建样本集：
X1 = np.array([[[1.2],[2.8]],
               [[1.9],[3.7]],
               [[2.5],[3.8]],
               [[4.8],[7.9]],
               [[5.6],[7.8]]])
X2 = np.array([[[9.7],[12.6]],
               [[10.8],[12.7]],
               [[13.7],[22.7]],
               [[7.48],[14.82]],
               [[11.23],[17.16]]])
N1 = X1.shape[0]
N2 = X2.shape[0]

#类均值向量：
m1 = np.array([[0],[0]])
for i in range(0,N1):
    m1 =m1+X1[i]
m1 = m1/N1
m2 = np.array([[0],[0]])
for i in range(0,N2):
    m2 =m2+X2[i]
m2 = m2/N2

#类内离散度矩阵：
S1 = np.zeros((2,2))
for i in range(0,N1):
    S1 = S1+np.dot(X1[i]-m1,np.transpose(X1[i]-m1))
S2 = np.zeros((2,2))
for i in range(0,N2):
    S2 = S2+np.dot(X2[i]-m2,np.transpose(X2[i]-m2))
Sw = S1+S2
#类间离散度矩阵：
Sb = np.dot(m1-m2,np.transpose(m1-m2))

#方向向量：
w = np.dot(np.linalg.inv(Sw),m1-m2)

print(w)


#投影后均值：
m11 = np.dot(np.transpose(w),m1)
m21 = np.dot(np.transpose(w),m2)
w0 = -(m11+m21)/2

#测试样本：
x_test = np.array([[3.42],[5.86]])
g = np.dot(np.transpose(w),x_test)+w0
if g>0:
    print('测试样本属于第一类！')
else:
    print('测试样本属于第二类！')


#可视化：
for i in range(0,N1):
    plt.scatter(X1[i,0],X1[i,1],c='r')
for i in range(0,N2):
    plt.scatter(X2[i,0],X2[i,1],c='b')
plt.scatter(x_test[0],x_test[1],c='g')
x = np.arange(0,15,0.01)
y = w[1]*x/w[0]
plt.plot(x,y,c='black')
plt.show()
