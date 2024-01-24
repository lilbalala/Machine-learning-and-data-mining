import numpy as np
import matplotlib.pyplot as plt
seed = np.random.seed(100)
#构造一个100行1列到矩阵。矩阵数值生成用rand，得到到数字是0-1到均匀分布到小数。
X = 2 * np.random.rand(100,1) #最终得到到是0-2均匀分布到小数组成到100行1列到矩阵。这一步构建列  X1(训练集数据)
#构建y和x的关系。 np.random.randn(100,1)是构建的符合高斯分布（正态分布）的100行一列的随机数。相当于给每个y增加列一个波动值。
y= 4 + 3 * X + np.random.randn(100,1)
#将两个矩阵组合成一个矩阵。得到的X_b是100行2列的矩阵。其中第一列全都是1.
X_b = np.c_[np.ones((100,1)),X]
#解析解求theta到最优解
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 生成两个新的数据点,得到的是两个x1的值
X_new = np.array([[0],[2]])
# 填充x0的值，两个1
X_new_b = np.c_[(np.ones((2,1))),X_new]
# 用求得的theata和构建的预测点X_new_b相乘，得到yhat
y_predice = X_new_b.dot(theta_best)
plt.plot(X_new, y_predice, 'r-')
# 画出已知数据X和掺杂了误差的y，用蓝色的点表示
plt.plot(X, y, 'b.')
# 建立坐标轴
plt.axis([0, 2, 0, 15, ])
plt.show()