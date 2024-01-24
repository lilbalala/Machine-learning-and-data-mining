import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

data = iris.data
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

newData = pca.fit_transform(data)
print(newData)