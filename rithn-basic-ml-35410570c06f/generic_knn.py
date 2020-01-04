"""
Created on Thu Sep 13 11:02:02 2018

@author: mayur.a
"""

import csv
import random
import math
import operator
import numpy as np
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def myreadCSVfile(split):
    replacement = {
            'HEALTHY' : 0,
            'MEDICATION' : 1,
            'SURGERY' : 2
            }
    med_data = pd.read_csv('Medical_data.csv').replace(replacement)
    #med_train, med_train_labels = np.array(med_data.values[:,1:],dtype=float),np.array(med_data[:,0],dtype=float)
    #print(med_train.shape) # Print Rows & Cols
    #print(med_train_labels.shape) # Print Rows & Cols
    med_data = np.array(med_data.values[1:], dtype=float)
    
    med_train, med_test = train_test_split(med_data,test_size=split)
    
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
    # Fill Nan values with 0
    rail_data.fillna(0, inplace=True)
    
	rail_data.to_csv('new_categorical_railway.csv')
	
    rail_data = np.array(rail_data.values[1:], dtype=int)
    rail_train, rail_test = train_test_split(rail_data,test_size=split)
     
    
    
    #import time
    #t=time.clock()
    chunksize = 50
    df_empty = pd.DataFrame()
    with open('fmnist-pca.csv') as f:
        chunk_iter = pd.read_csv(f, header=None, chunksize=chunksize)
        for chunk in chunk_iter:
            df_empty = pd.concat([df_empty,chunk])
            
    #print(df_empty)
    #print(df_empty.shape)
    #print("Time taken in loading train data: " , time.clock()-t)
    mnist_data = np.array(df_empty.values[0:], dtype=float)
    #print(mnist_data)
    mnist_data = np.transpose(mnist_data)
    #print(mnist_data)
    #amount = 5000
    #indices = np.random.random_sample(size=amount)
    #print(indices)
    mnist_train, mnist_test = train_test_split(mnist_data,test_size=split)
    
    return med_train, med_test, rail_train, rail_test, mnist_train, mnist_test

def euclidean_dist(pt1, pt2, nof):

    dist = 0

    for i in range(1,nof):

        dist += pow((pt1[0][i] - pt2[0][i]), 2)

    return math.sqrt(dist)


def getN(traindata, testCase, k):

    dists = []

    #length = len(testCase)-1

    r,c = testCase.shape
    #length = len(testCase)
    #print(traindata)
    #for i in range(len(traindata)):
    for i in range(len(traindata)):

        dist = euclidean_dist(traindata[i:i+1], testCase, c)

        dists.append((traindata[i:i+1], dist))

    dists.sort(key=operator.itemgetter(1))

    neighbors = []

    for i in range(k):

        neighbors.append(dists[i][0])

    return neighbors

def getR(nn):
    class_V = {}
    for i in range(len(nn)):
        x = np.array(nn[i][0])
        response = x[0]
        #response = nn[i][0]
        if response in class_V:
            class_V[response] += 1
        else:
            class_V[response] = 1
    sortedV = sorted(class_V.items(), key=operator.itemgetter(1), reverse=True)
    return sortedV[0][0]

def get_accuracy(testdata, predictn):
    correct = 0
    cnt_fp = 0  # count False Postive
    for i in range(len(testdata)):
        if testdata[i][0] == predictn[i]:
            correct += 1    # True Positive
        else:
            if testdata[i][1] == 1:
                cnt_fp += 1
        #print("class: " + predictn[i] + " test_data: " + testdata[i][0])
        #print('> predicted class =' + repr(predictn[i]) + ',   actual testdata =' + repr(testdata[i][0]))
    return ((correct)/float (len(testdata)))*100, cnt_fp

def main():
    # Define arrays to get training & test data from csv reader function
    m_train = []
    m_test = []
    t_train = []
    t_test = []
    mnist_train = []
    mnist_test = []
    
    spiltratio = 0.3
    # Load file & prepare data, split it inot 70% and 30% ratio

    #x_train, x_test = LoadDataSet('Medical_data.csv', spiltratio, x_train, y_train, x_test, y_test)

    #x_train, x_test = LoadDataSet('Medical_data.csv', spiltratio)
    m_train, m_test, t_train, t_test, mnist_train, mnist_test = myreadCSVfile(spiltratio)

#Test 1    
    # Accuracy for Medical test
    #x_train = m_train
    #x_test = m_test
#Test 2
    # Accuracy for RailwayBooking
    #x_train = t_train
    #x_test = t_test
#Test 3
    # Accuracy for Fashion mnist
    
    x_train = mnist_train
    x_test = mnist_test
    
    # Print the train data and test data
    print ('Train data: ' + repr(len(x_train)))
    print ('Test data : ' + repr(len(x_test)))

    for k in range (60,500):
    
        predictions = []
        #k=5
        print("for K = ",k)
        for i in range(len(x_test)):
    
            #print(x_train)
            #print(i)
            neighbors = getN(x_train, x_test[i:i+1], k)
            result = getR(neighbors)
            predictions.append(result)
        accuracy,cnt_fp = get_accuracy(x_test, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    
    """
    In case of Railbooking List, there is 0/1 classifier, so we will calculate 
    Confusion Matrix
    """
    """
    TP = np.count_nonzero(predictions) # TP
    TN = len(predictions) - TP  # TN
    #print(cnt)
    FP = cnt_fp
    FN = TP - TN - FP
    Recall = (TP/(TP+FN))*100
    print('Recall: ' + repr(Recall) + '%')
    Precision = (TP/(TP+FP))*100
    print('Precision: ' + repr(Precision) + '%')
    F1_Measure = ((2*Recall*Precision)/(Recall+Precision))
    print('F1_Measure: ' + repr(F1_Measure) + '%')
    """
main()