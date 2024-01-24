import matplotlib.pyplot as plt
import numpy as np
import random

def distance(x,y):
    z = np.expand_dims(x, axis=1) - y
    z = np.square(z)
    z = np.sqrt(np.sum(z, axis=2))
    return z

def k_means(data, k, max_iter=20):
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_features = data.shape
    # 随机初始化簇中心
    indices = random.sample(range(n_samples), k)
    center = np.copy(data[indices])
    cluster = np.zeros(data.shape[0], dtype=np.int32)
    i = 1
    while i <= max_iter:
        dis = distance(data, center)
        # 样本新的所属簇
        cluster = np.argmin(dis, axis=1)
        onehot = np.zeros(n_samples * k, dtype=np.float32)
        onehot[cluster + np.arange(n_samples) * k] = 1.
        onehot = np.reshape(onehot, (n_samples, k))
        new_center = np.matmul(np.transpose(onehot, (1, 0)), data)
        new_center = new_center / np.expand_dims(np.sum(onehot, axis=0), axis=1)
        center = new_center
        i += 1
    return cluster, center

def scatter_cluster(data, cluster, center):
    if data.shape[1] != 2:
        raise ValueError('Only can scatter 2d data!')
    # 画样本点
    plt.scatter(data[:, 0], data[:, 1], c=cluster, alpha=0.8)
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '&lt;b', 'pb']
    # 画质心点
    for i in range(center.shape[0]):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize=20)
    plt.show()

n_samples = 500
n_features = 2
k = 3
data = np.random.randn(n_samples, n_features)
cluster, center = k_means(data, k)
scatter_cluster(data, cluster, center)