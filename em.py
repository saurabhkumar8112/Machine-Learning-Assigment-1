import numpy as np
import pandas as pd
import math

# k is number of gaussians, d is dimension of datapt
def em_for_gmm(dataset, k):
    n, d = dataset.shape 
    omega = np.random.rand(n,k) 
    print(n)
    omega = omega / (np.sum(omega, axis = 1).reshape(n,1))
    adiff = 1
    alphaprev = np.full((1, k), 0.0)
    while (adiff > 0.00001):
        alphas = np.sum(omega, axis = 0) / float(n)
        # print(omega)
        means = np.array([np.sum(dataset*(omega[:,j].reshape(n,1)), axis = 0) for j in range(k)])
        # print(dataset*(omega[:,0].reshape(n,1)))
        # print(dataset[:,4])
        # print(means)
        means = means/ (n*alphas.reshape(k,1))
        
        vars = np.array([np.sum([np.diagonal(np.dot((dataset[i] - means[j]).reshape(d, 1), (dataset[i] - means[j]).reshape(1,d)))*omega[i][j] for i in range(n)], axis = 0) for j in range(k)])
        vars = vars/(n*alphas.reshape(k,1))
        vars+=np.random.rand(k,d)*0.000001
        for i in range(n):
            for j in range(k):
                omega[i][j] = math.exp(-0.5*np.dot(((dataset[i]-means[j])/vars[j]), np.transpose(dataset[i]-means[j])))
                omega[i][j] /= (np.prod(vars[j])*((2*math.pi)**d))**0.5
                omega[i][j] *= alphas[j]
        print(vars)
        omega = omega / np.sum(omega, axis = 1).reshape(n,1)
        adiff = np.linalg.norm(alphaprev-alphas)
        alphaprev = alphas
        print (adiff, alphaprev)

replacements = {
        'FIRST_AC': 1,
        'SECOND_AC': 2,
        'THIRD_AC': 3,
        'NO_PREF': 4,
        'male' : 1,
        'female' : 0
        }

data = pd.read_csv('railwayBookingList.csv').replace(replacements)
data = data.drop(columns=['caseID'])
data=np.array(data)		
em_for_gmm(data[:,:], 8)