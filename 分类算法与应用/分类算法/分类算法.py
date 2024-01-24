import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# 加载数据
input_file = 'data/adult.data.txt'
X = []
y = []
num_lessthan50k = 0
num_morethan50k = 0
num_threshold = 30000
with open(input_file,'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50k' and num_lessthan50k < num_threshold:
            X.addpend(data)
            num_lessthan50k = num_lessthan50k + 1
        elif data[-1] == '>50k' and num_morethan50k <num_threshold:
            X.append(data)
            num_morethan50k = num_morethan50k + 1
        if num_lessthan50k >= num_threshold and num_morethan50k >= num_threshold:
            break
X = np.array(X)

#转换数据的属性编码，因为原数据中的属性是包含英文字符，取法进行数学运算，请将其转化为数值类型数据
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        label_encoder.append(le)
        # 转换数据的属性，已知原数据中含有英文字符，请将英文转换为数值型，使用preprocessing.LabelEncoder()函数
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# 创建分类器并进行训练并进行交叉验证
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)
# 计算F1得分
f1 = model_selection.cross_val_score(classifier_gaussiannb,X, y, scoring='f1_weighted', cv=5)
print ("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# 创建个例，将其进行同样编码处理
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count = 0
input_data_encoded = [-1] * len(input_data)
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1
input_data_encoded = np.array(input_data_encoded)

#将个体进行预测分类，并输出结果
output_class = classifier_gaussiannb.predict(input_data_encoded.reshape(1,-1))
print (label_encoder[-1].inverse_transform(output_class)[0])