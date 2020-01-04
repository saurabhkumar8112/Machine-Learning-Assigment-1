# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:30:05 2018

@author: mayur.a
"""
import csv
import random
import math
import operator
    
def LoadDataSet(filename, split, traindata, testdata):
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile)
        next(data)
        dataset = list(data)
        for x in range(len(dataset)-1):
            for y in range(1,4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                traindata.append(dataset[x])
            else:
                testdata.append(dataset[x])

#nof = no of features/length
def euclidean_dist(pt1, pt2, nof):
    dist = 0
    for i in range(1, nof):
        dist += pow((pt1[i] - pt2[i]), 2)
    return math.sqrt(dist)

def getN(traindata, testCase, k):
    dists = []
    length = len(testCase)-1
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
        response = nn[i][0]
        if response in class_V:
            class_V[response] += 1
        else:
            class_V[response] = 1
    sortedV = sorted(class_V.items(), key=operator.itemgetter(1), reverse=True)
    return sortedV[0][0]

def get_accuracy(testdata, predictn):
    correct = 0
    for i in range(len(testdata)):
        if testdata[i][0] == predictn[i]:
            correct += 1
        #print("class: " + predictn[i] + " test_data: " + testdata[i][0])
        #print('> predicted class =' + repr(predictn[i]) + ',   actual testdata =' + repr(testdata[i][0]))
    return ((correct)/float (len(testdata)))*100
 

def main():
    trainData = []
    testData = []
    spiltratio = 0.8
    # Load file & prepare data, split it inot 70% and 30% ratio
    LoadDataSet('Medical_data.csv', spiltratio, trainData, testData)
    # Print the train data and test data
    print ('Train data: ' + repr(len(trainData)))
    print ('Test data : ' + repr(len(testData)))
    # generate prediction and find the accuracy based on K = 5
    #for k in range(3, 7):    
    for k in range(1, 30):
        predictions = []
        for i in range(len(testData)):
            neighbors = getN(trainData, testData[i], k)
            result = getR(neighbors)
            predictions.append(result)
        accuracy = get_accuracy(testData, predictions)
        print('Accuracy: ' + repr(accuracy) + '%'+" "+str(k))
#    y_accuracy = int((accuracy*len(testData))/100)
#   precision = get_precision(y_accuracy)
#   recall = get_recall(y_accuracy)
#  fscore = get_fscore(y_accuracy)
#   print('precision: {}'.format(precision))
#   print('recall: {}'.format(recall))
#   print('fscore: {}'.format(fscore))
    
main()
