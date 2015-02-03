#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)
print len(enron_data["SKILLING JEFFREY K"])

count = 0
names = []

for name, features in enron_data.items():
	top_feature = enron_data[name]["total_payments"]
	poi_feature = enron_data[name]["poi"]
	if poi_feature == True and top_feature == 'NaN':
		count+=1
	names.append(name)

print count


'''
skilling = enron_data["SKILLING JEFFREY K"]["total_payments"]
lay = enron_data["LAY KENNETH"]["total_payments"]
fastow = enron_data["FASTOW ANDREW"]["total_payments"]

print "Lay: ", lay
print "Skilling: ", skilling
print "Fastow: ", fastow 
'''
