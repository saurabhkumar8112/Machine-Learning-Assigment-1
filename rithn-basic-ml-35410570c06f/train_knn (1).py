# -*- coding: utf-8 -*-
"""
Created on Fri Sep  10 15:15:05 2018

@author: mayur.a
"""
import csv
import random
import math
import operator
    
def LoadDataSet(filename, split, traindata, testdata):
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile)
        
        dataset = list(data)
        
        for x in range(1, len(dataset)):
            for y in range(3,7):
                if y==4:
                    if dataset[x][y]=='FIRST_AC':
                        dataset[x][y] = '1'
                    if dataset[x][y]=='SECOND_AC':
                        dataset[x][y] = '2'      
                    if dataset[x][y]=='THIRD_AC':
                        dataset[x][y] = '3'
                    if dataset[x][y]=='NO_PREF':
                        dataset[x][y] = '4'
                if y==5:
                    if dataset[x][y]=='male':
                        dataset[x][y] = '0'
                    elif dataset[x][y]=='female':
                        dataset[x][y] = '1'      
                    else:
                        dataset[x][y] = '0'
        for x in range(1, len(dataset)-1):
            for y in range(0,7):
                dataset[x][y] = int(dataset[x][y])
            if random.random() < split:
                traindata.append(dataset[x])
            else:
                testdata.append(dataset[x])

def euclidean_dist(pt1, pt2, nof):
    dist = 0
    for i in range(2, nof):
        dist += pow((pt1[i] - pt2[i]), 2)
    return math.sqrt(dist)

def getN(traindata, testCase, k):
    dists = []
    length = len(testCase)
    #print(traindata)
    for i in range(len(traindata)):
        dist = euclidean_dist(traindata[i], testCase, length)
        dists.append((traindata[i], dist))
    dists.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(dists[i][0])
    return neighbors

def getR(nn):
    class_V = {}
    for i in range(len(nn)):
        response = nn[i][1]
        if response in class_V:
            class_V[response] += 1
        else:
            class_V[response] = 1
    sortedV = sorted(class_V.items(), key=operator.itemgetter(1), reverse=True)
    return sortedV[0][0]

def get_accuracy(testdata, predictn):
    correct = 0
    for i in range(len(testdata)):
        if testdata[i][1] == predictn[i]:
            correct += 1
        #print("class: " + predictn[i] + " test_data: " + testdata[i][1])
        #print('> predicted class =' + repr(predictn[i]) + ',   actual testdata =' + repr(testdata[i][0]))
    return ((correct)/float (len(testdata)))*100

def main():
    trainData = []
    testData = []
    spiltratio = 0.7
    # Load file & prepare data, split it inot 70% and 30% ratio
    LoadDataSet('railwayBookingList.csv', spiltratio, trainData, testData)
    # Print the train data and test data
    print ('Train data: ' + repr(len(trainData)))
    print ('Test data : ' + repr(len(testData)))
    # generate prediction and find the accuracy based on K = 8
    #for k in range(4,15):
    predictions = []
    k = 8
    for i in range(len(testData)):
        neighbors = getN(trainData, testData[i], k)
        result = getR(neighbors)
        predictions.append(result)
   
    accuracy = get_accuracy(testData, predictions)
    #print('For k = :',k)
    print('Accuracy: ' + repr(accuracy) + '%')
main()

