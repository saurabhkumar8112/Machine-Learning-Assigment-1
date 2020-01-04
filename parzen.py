"""
	we can investigate if it is better to compute a parzen window for n-variables and then 
	sum them over other axes to get marginal distribution of each feature, or to do Parzen window on 
	each vector separately.
	
	This file contains a function to convert a numpy array into a probability distribution.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Here arr is expected to be n*d array with n datapoints having d dimensions. 
# If it is a 1-d array, do arr.reshape(1,d) to ensure smooth functioning
# x is d-dimensional
# this function outputs an estimate of the p.d.f. at x given dataset arr and size of window determined by h.
def parzen_pdf(x, h, arr):
	n, d = arr.shape
	intmdt = sum((math.exp(-1*(np.linalg.norm(x - arr[i])**2)/(n*h*h)) for i in range(n)))
	return intmdt / (n*((2*math.pi*h)**d))
	
def parzen_pdf2(x, h, arr):
	n, d = arr.shape
	ct = 0
	for pt in arr:
		for i in range(d):
			if abs(pt[i]-x[i]) > h:
				break
		else:
			ct += 1
	return ct
			
# Here training data is n*d and datapt is 1*d	
def parzen_classifier(training_data, training_labels, datapt, categories, h0,param):
    ccd = []
    for category in categories:
        ccdataset = obtain_ccdata(training_data, training_labels, category)
        if(param==1):
            ccd.append(parzen_pdf(datapt, h0, ccdataset)*len(ccdataset))
        else:
            ccd.append(parzen_pdf2(datapt, h0, ccdataset)*len(ccdataset))
        
    return categories[ccd.index(max(ccd))]

def obtain_ccdata(training_data, training_labels, category):
	n, d = training_data.shape
	return np.array([training_data[i] for i in range(n) if training_labels[i] == category])
	
categories1 = ["HEALTHY", "MEDICATION", "SURGERY"]
categories2 =[0,1]
categories3=[0,1,2,3,4,5,6,7,8,9]
replacements = {
            'FIRST_AC': 1,
            'SECOND_AC': 2,
            'THIRD_AC': 3,
            'NO_PREF': 4,
            'male' : 1,
            'female' : 0
            }

rail_data = pd.read_csv('railwayBookingList.csv').replace(replacements)
rail_data = rail_data.drop(columns=['caseID'])
df = pd.read_csv("Medical_data.csv")

for i in pd.read_csv('fmnist-pca.csv', chunksize = 25, header=None):
        data = i
        break
arr = np.array(data)
arr=arr[:,:1000]
arr=arr.T
n, d = arr.shape
print(arr.shape)
np.random.shuffle(arr)
tlabels, tdata, tstlabels, tstdata = arr[:int(0.8*n),0], arr[:int(0.8*n),1:], arr[int(0.8*n):,0], arr[int(0.8*n):,1:]
accu1=[]
accu2=[]
h_val=[0.001,0.003,0.006, 0.01, 0.02, 0.03, 0.1,0.3,0.6,1.0,2.0,4.0,6.0,10.0]
for i in range(1,3):
    for h in [0.001,0.003,0.006, 0.01, 0.02, 0.03, 0.1,0.3,0.6,1.0,2.0,4.0,6.0,10.0]:
        results = [parzen_classifier(tdata, tlabels, pt, categories3, h,i) for pt in tstdata]
        accuracy = sum(map(int, [tstlabels[i] == results[i] for i in range(len(tstdata))]))/float(len(tstdata))
        accuracy*=100
        if(i==1):
            accu1.append(accuracy)
        else:
            accu2.append(accuracy)
        print(accuracy)
    if(i==1):
        plt.plot(np.log10(h_val),accu1,'r')
        plt.legend("For Normal Distribution") 
        plt.ylabel('Accuracy')
        plt.xlabel('log(h)')
        plt.grid(b=True)
    else:
        plt.plot(np.log10(h_val),accu2,'g')
        plt.legend("For Hyper Cube")
        plt.ylabel('Accuracy')
        plt.xlabel('log(h)')
        plt.grid(b=True)
plt.show()