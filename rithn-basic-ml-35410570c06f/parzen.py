"""
	we can investigate if it is better to compute a parzen window for n-variables and then 
	sum them over other axes to get marginal distribution of each feature, or to do Parzen window on 
	each vector separately.
	
	This file contains a function to convert a numpy array into a probability distribution.
"""
import math
import numpy as np
import pandas as pd


# Here arr is expected to be n*d array with n datapoints having d dimensions. 
# If it is a 1-d array, do arr.reshape(1,d) to ensure smooth functioning
# x is d-dimensional
# this function outputs an estimate of the p.d.f. at x given dataset arr and size of window determined by h.
def parzen_pdf(x, h, arr):
	n, d = arr.shape
	intmdt = sum((math.exp(-1*(np.linalg.norm(x - arr[i])**2)/(n*h*h)) for i in range(n)))
	return intmdt / (n*((2*math.pi*h)**d))

# Here training data is n*d and datapt is 1*d	
def parzen_classifier(training_data, training_labels, datapt, categories, h0):
	ccd = []
	for category in categories:
		ccdataset = obtain_ccdata(training_data, training_labels, category)
		ccd.append(parzen_pdf(datapt, h0, ccdataset)*len(ccdataset))
	return categories[ccd.index(max(ccd))]
	
def obtain_ccdata(training_data, training_labels, category):
	n, d = training_data.shape
	return np.array([training_data[i] for i in range(n) if training_labels[i] == category])
	
categories = ["HEALTHY", "MEDICATION", "SURGERY"]
df = pd.read_csv("Medical_data.csv")
arr = np.array(df)
labels, data = arr[:,0], arr[:,1:]
for h in [0.001, 0.06, 0.01, 0.1, 0.3]:
	results = [parzen_classifier(data, labels, pt, categories, h) for pt in data]
	accuracy = sum(map(int, [labels[i] == results[i] for i in range(len(data))]))/float(len(data))
	print (accuracy)
