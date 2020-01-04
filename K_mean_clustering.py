import pandas as pd
import numpy as np
import math
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
def euc_dis_min(dataset,point):
  dist=[]
  for i in range(len(point)):
    dis=0
    for j in range(len(dataset)):
      dis+=math.pow(dataset[j]-point[i][j],2)
    dist.append(math.sqrt(dis))
  return dist.index(min(dist))

#select k random points
def k_init(k,lim):
  return np.random.randint(0,lim,size=(k))

#the algorithm goes here
def k_mean(dataset,k):
    x,y=dataset.shape   #x is the number of data point and y is the dimension
    centroids_now=dataset[k_init(k,x)]  #getting the initial random centroids
    # print(centroids_now)
    cluster=[0]*x
    # print cluster
    error=1000 #say
    tolerance=1
    iter=4
    while(iter>1 and error>1):
        for i in range(x):
            cluster[i]=(euc_dis_min(dataset[i],centroids_now))
        # print cluster
        # print type(cluster[0])
        #taking mean as the new centroid
        centroids_copy=deepcopy(centroids_now)
        # print (centroids_copy, "\n", centroids_now)
        for i in range(k):
            points=[dataset[j] for j in range(x) if cluster[j]==i]
            # print(points)
            if len(points) != 0:
                # print i
                centroids_now[i]=np.mean(points,axis=0)
        error=sq_error(centroids_copy,centroids_now)
        print("im working")
        iter-=1
        # print(error)
    return cluster, centroids_now

file_name1='railwayBookingList.csv'
file_name2='fmnist-pca.csv'
data=pd.read_csv(file_name2)
data=np.array(data)
data=data[:100,:]
data=data.T
x,y=data.shape
print(data.shape)

	      
errs = []
for k in range(1, 80):
    assign, c=k_mean(data[:,1:],k)
    errs.append(sum([np.linalg.norm(data[i][1:]-c[assign[i]]) for i in range(x)]))
    print (errs[-1])
plt.plot(errs)
plt.show()