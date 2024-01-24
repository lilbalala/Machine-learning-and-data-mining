import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    housing_data = fetch_california_housing()

    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt)

    print("#### 决策树性能 ####")
    print("均方误差 =", round(mse, 2))
    print("可解释方差分 =", round(evs, 2))

    feature_importances = 100.0 * (dt_regressor.feature_importances_ / max(dt_regressor.feature_importances_))

    index_sorted = np.flipud(np.argsort(feature_importances))

    pos = np.arange(index_sorted.shape[0]) + 0.5
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, index_sorted)  # 修改这一行
    plt.ylabel('相对重要性')
    plt.title('决策树回归器')
    plt.show()
