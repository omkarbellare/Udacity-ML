#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
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



### it's all yours from here forward!  

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
