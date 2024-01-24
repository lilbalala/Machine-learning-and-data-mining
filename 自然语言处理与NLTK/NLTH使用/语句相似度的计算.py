data = ["This two-wheeler is really good on slippery rodes"]
sentce = ["This is really good"]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)
print("\nTfidf of training data:",X_train_tfidf.toarray())

X_input_termcounts = vectorizer.transform(sentce)
X_input_tfdf = tfidf_transformer.transform(X_input_termcounts)
print("\nTfidf of traing data:",X_input_tfdf.toarray())

print("\nCosine of data:",cosine_similarity(X_train_tfidf,X_input_tfdf))