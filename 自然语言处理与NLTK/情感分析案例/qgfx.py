import random
import numpy as np
import csv
import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

file_path = 'data/review.csv'
userdict = 'data/userdict.txt'
stopword_path = 'data/stopwords.txt'
jieba.load_userdict(userdict)

def load_file_to_list(file_path):
    with open(file_path, 'r',encoding='utf8') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    review_data = np.array(rows).tolist()
    random.seed(100)
    random.shuffle(review_data)

    review_list = []
    sentiment_label_list = []
    for words in review_data:
        review_list.append(words[1])
        sentiment_label_list.append(words[0])
    return review_list, sentiment_label_list

review_list, sentiment_label_list = load_file_to_list(file_path)

print('review_list[:10]:',review_list[:10])
print('sentiment_label_list[:10]:',sentiment_label_list[:10])
print('='*30)

train_review_list,test_review_list,train_sentiment_label_list,test_sentiment_label_list= train_test_split(review_list,sentiment_label_list,test_size=0.2,random_state=0)
print('Numbers of train sets： {}'.format(str(len(train_review_list))))
print('Number of test sets： {}'.format(str(len(test_review_list))))
print('='*30)

def load_stopwords(file_path):
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
       stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words

def review_to_text(review):
    stop_words = load_stopwords(stopword_path)
    review = re.sub("[^\u4e00-\u9fa5^a-z^A-Z]", '', review)
    review = jieba.cut(review)
    if stop_words:
        all_stop_words = set(stop_words)
        words = [w for w in review if w not in all_stop_words]

    return words

review_train = [' '.join(review_to_text(review)) for review in train_review_list]
sentiment_label_train = train_sentiment_label_list

review_test = [' '.join(review_to_text(review)) for review in test_review_list]
sentiment_label_test = test_sentiment_label_list

count_vec = CountVectorizer(max_df=0.8, min_df=3)

def bayes_Classifier():
    return Pipeline([
        ('count_vec', CountVectorizer()),
        ('mnb', MultinomialNB())
    ])
bayes_clf = bayes_Classifier()

bayes_clf.fit(review_train, sentiment_label_train)

print('test set accuracy： {}'.format(bayes_clf.score(review_test, sentiment_label_test)))

strpre = '无论看多少遍，都不懂欣赏。。'
strpre1 = '很好看的一部电影啊'
prelabel = bayes_clf.predict( [' '.join(review_to_text(strpre))])
prelabel1 = bayes_clf.predict( [' '.join(review_to_text(strpre1))])
print(strpre,':',prelabel)
print(strpre1,':',prelabel1)
print('='*30)

