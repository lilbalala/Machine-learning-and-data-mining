# 导包
from sklearn import datasets
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 拆分鸢尾花数据集
data=iris.data
from sklearn.decomposition import FactorAnalysis
# 实例化估计器
fa=FactorAnalysis(n_components=2)
# 训练数据
newData1=fa.fit_transform(data)
#　打印输出
print(newData1)