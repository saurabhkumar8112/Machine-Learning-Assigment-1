import pandas as pd
import numpy as np
import math
import time
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#total error distance after centroid have fixed for a K
def sq_error(dataset,point):
  x,y=dataset.shape
  dist=0
  for i in range(x):
    dist+=np.linalg.norm(dataset[i]-point[i])
  return dist
 
#minimum eucleidan distance calcullation
def euc_dis_min(datapt,centers):
  dist=[]
  for i in range(len(centers)):
    dist.append(np.linalg.norm(datapt - centers[i]))
  return dist.index(min(dist))

#select k random points
def k_init(k,lim):
  return list(range(k)) 

#the algorithm goes here
def k_mean(dataset,k):
    x,y=dataset.shape   #x is the number of data point and y is the dimension
    centroids_now=dataset[k_init(k,x)]  #getting the initial random centroids
    # print(x,y)
    # print(centroids_now)
    cluster=[0]*x
    # print cluster
    error=1000 #say
    tolerance=1
    iter=4
    while(iter>1 and    error>tolerance):
        for i in range(x):
            cluster[i]=(euc_dis_min(dataset[i],centroids_now))
        #taking mean as the new centroid
        centroids_copy=deepcopy(centroids_now)
        for i in range(k):
            points=[dataset[j] for j in range(x) if cluster[j]==i]
            # print(points)
            if len(points) != 0:
                centroids_now[i]=np.mean(points,axis=0)
        error=sq_error(centroids_copy,centroids_now)
        print("im working")
        iter-=1
        
    return cluster, centroids_now

m_file='Medical_data.csv'
t_file='railwayBookingList.csv'
fmnist_file='fmnist-pca.csv'
for i in pd.read_csv(fmnist_file, chunksize = 50, header=None):
	data_fmnist = i
	break
data_fmnist=np.array(data_fmnist)
data_fmnist=data_fmnist[:,:5000]
data_fmnist=data_fmnist.T
replacements = {
            'FIRST_AC': 1,
            'SECOND_AC': 2,
            'THIRD_AC': 3,
            'NO_PREF': 4,
            'male' : 1,
            'female' : 0
            }
data_rail=pd.read_csv(t_file).replace(replacements)
data_rail = data_rail.drop(columns=['caseID'])
data_med=pd.read_csv(m_file)
for i in range(3,4):
    if(i==1):
        data=data_med
        print("MEDICAL DATA")
    elif(i==2):
        data=data_rail
        print("RAILWAY DATA")
    else:
        data=data_fmnist
        print("FMNIST DATA")
    data=np.array(data)
    print(data.shape)
    x,y=data.shape
       
    errs = []
    with open('log.txt','a+') as f:
        f.write("For "+str(i))
    for k in range(1, 20):
        assign, c=k_mean(data[:,1:],k)
        errs.append(sum([np.linalg.norm(data[i][1:]-c[assign[i]]) for i in range(x)]))
        print (errs[-1])
        with open('log.txt','a+') as f:
            f.write(str(k)+": "+str(errs[-1])+"\n")
    
    if(i==1):
        plt.plot(errs,'r')
        plt.title('Error Vs Number of Clusters for Medical Data')
    elif(i==2):
        plt.plot(errs,'b')
        plt.title('Error Vs Number of Clusters for railwayBooking Data')
    elif(i==3):
        plt.plot(errs,'g')
        plt.title('Error Vs Number of Clusters for FMNIST Data')
    plt.ylabel('Error')
    plt.xlabel('Number of Clusters')
    plt.grid(b=True)
    plt.show()