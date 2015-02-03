#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

#Split the data into training and testing
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
			features, labels, test_size = 0.3, random_state = 42)

#Classify using a Decision Tree
from sklearn import tree

#Create a basic DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

#Fit the DT to the training set with training labels
clf = clf.fit(features_train, labels_train)

#Predict the training labels
predTrain = clf.predict(features_train)

#Predict the testing labels
predTest = clf.predict(features_test)

#Print out the accuracy score
from sklearn.metrics import accuracy_score
accScore = accuracy_score(labels_train, predTrain)
print "Accuracy on training set: %0.2f" % (accScore)

accScore = accuracy_score(labels_test, predTest)
print "Accuracy on testing set: %0.2f" % (accScore)

countPOI = 0
truePositives = 0
falseNegatives = 0
falsePositives = 0
for i in range(len(predTest)):
	if predTest[i] == 1:
		countPOI += 1
	if predTest[i] == 1 and labels_test[i] == 1:
		truePositives += 1
	elif predTest[i] == 1 and labels_test[i] == 0:
		falsePositives += 1
	elif predTest[i] == 0 and labels_test[i] == 1:
		falseNegatives += 1

print "Number of POIs predicted : %d" % (countPOI)
print "Actual number of POIs : %d" % sum(labels_test)

print ""

print "True Positives : %d" % (truePositives)
print "False Positives : %d" % (falsePositives)
print "False Negatives : %d" % (falseNegatives)

print ""

from sklearn.metrics import precision_score, recall_score
precision = precision_score(labels_test, predTest)
print "Precision : %0.3f" % (precision)

recall = recall_score(labels_test, predTest)
print "Recall : %0.3f" % (recall)

print "Number of people in test set : %d" % len(predTest)
