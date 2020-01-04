import numpy as np
import pandas as pd
import math

# k is number of gaussians, d is dimension of datapt
def em_for_gmm(dataset, k):
	n, d = dataset.shape 
	omega = np.random.rand(n,k) 
	omega = omega / (np.sum(omega, axis = 1).reshape(n,1))
	adiff = 1
	alphaprev = np.full((1, k), 0.0)
	while adiff > 0.00001:
		alphas = np.sum(omega, axis = 0) / float(n)
		means = np.array([np.sum(dataset*(omega[:,j].reshape(n,1)), axis = 0) for j in range(k)])
		means = means/ (n*alphas.reshape(k,1))
		vars = np.array([np.sum([np.diagonal(np.dot((dataset[i] - means[j]).reshape(d, 1), (dataset[i] - means[j]).reshape(1,d)))*omega[i][j] for i in range(n)], axis = 0) for j in range(k)])
		vars = vars/(n*alphas.reshape(k,1))
		for i in range(n):
			for j in range(k):
				omega[i][j] = math.exp(-0.5*np.dot(((dataset[i]-means[j])/vars[j]), np.transpose(dataset[i]-means[j])))
				omega[i][j] /= (np.prod(vars[j])*((2*math.pi)**d))**0.5
				omega[i][j] *= alphas[j]
		omega = omega / np.sum(omega, axis = 1).reshape(n,1)
		adiff = np.linalg.norm(alphaprev-alphas)
		alphaprev = alphas
		print adiff, alphaprev
		
data = np.array(pd.read_csv("Medical_data.csv"))
em_for_gmm(data[:, 1:], 6)