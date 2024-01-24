from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

dig = datasets.load_digits()
X_train,X_test,y_train,y_test = train_test_split(dig.data,dig.target,test_size=0.4,random_state=0)

clf = DecisionTreeClassifier()
rfc = RandomForestClassifier()
rfc2 = RandomForestClassifier(n_estimators=200,max_features=8)
clf_pre = clf.fit(X_train, y_train).predict(X_test)
rfc_pre =rfc.fit(X_train, y_train).predict(X_test)
rfc2_pre =rfc2.fit(X_train, y_train).predict(X_test)

print(metrics.accuracy_score(y_test, clf_pre))
print(metrics.accuracy_score(y_test, rfc_pre))
print(metrics.accuracy_score(y_test, rfc2_pre))