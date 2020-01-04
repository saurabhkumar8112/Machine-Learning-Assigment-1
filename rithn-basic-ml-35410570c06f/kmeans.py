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
    print(x,y)
    # print(centroids_now)
    cluster=[0]*x
    # print cluster
    error=1000 #say
    tolerance=1
    iter=4
    while(iter>1 and error>1):
        t0 = time.time()
        for i in range(x):
            cluster[i]=(euc_dis_min(dataset[i],centroids_now))
        t1 = time.time()
        # print cluster
        # print type(cluster[0])
        #taking mean as the new centroid
        centroids_copy=deepcopy(centroids_now)
        t2 = time.time()
        # print (centroids_copy, "\n", centroids_now)
        for i in range(k):
            points=[dataset[j] for j in range(x) if cluster[j]==i]
            # print(points)
            if len(points) != 0:
                # print i
                centroids_now[i]=np.mean(points,axis=0)
        t3 = time.time()
        error=sq_error(centroids_copy,centroids_now)
        print("im working")
        iter-=1
        t4 = time.time()
        print(t1- t0, t2-t1, t3-t2, t4-t3)
        # print(error)
    return cluster, centroids_now

file_name='fmnist-pca.csv'
for i in pd.read_csv(file_name, chunksize = 101, header=None):
	data = i
	break
data=np.array(data)
data=data.T
x,y=data.shape
print(data.shape)

	      
errs = []
for k in range(1, 30):
    assign, c=k_mean(data[:,1:],k)
    errs.append(sum([np.linalg.norm(data[i][1:]-c[assign[i]]) for i in range(x)]))
    print (errs[-1])
    with open('log.txt','a+') as f:
        f.write(str(k)+": "+str(errs[-1])+"\n")
plt.plot(errs)
plt.show()