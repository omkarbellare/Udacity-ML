#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
#words_file = "word_data_overfit.pkl" ### like the file you made in the last mini-project 
words_file = "../text_learning/your_word_data.pkl"
#authors_file = "email_authors_overfit.pkl"  ### this too
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (remainder go into training)
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

featureImportance = clf.feature_importances_

maxImp = 0.0
idx = 0
countHigh = 0
print len(features_train), len(features_train[0])
for i in range(len(featureImportance)):
	if featureImportance[i] > 0.2:
		countHigh += 1
	if featureImportance[i] > maxImp:
		maxImp = featureImportance[i]
		idx = i

print "Value: %f, Index: %d" % (maxImp, idx)

print vectorizer.get_feature_names()[idx]

print "Count of features with importance > 0.2 : %d " % (countHigh)
