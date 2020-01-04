
"""
	Download folders data and utils from git link of fashion-mnist dataset mentioned in assignment.

	Put this file at same level of nesting i.e.
	|
	|--pca.py
	|--data
	|--utils
	
	NUM is number of samples <= 60000
	LIMIT is number of eigenvectors to be considered <= 784
	
	testing shows 20-50 eigenvectors are suitable, find them in order by eigenvalues in evls with index for evcts
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import mnist_reader 

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train_mn = np.mean(x_train, axis=0)
x_train_n = x_train - x_train_mn
corr = np.dot(np.transpose(x_train_n), x_train_n)

evls, evcts = np.linalg.eig(corr)
evls = sorted(enumerate(evls), key = lambda x:x[1], reverse = True)
errs = np.empty(784)
errs.fill(0.0)

print("Loaded everything!")

NUM = 60000
LIMIT = 100

for i in range(NUM):
	l = x_train[i]
	if (i%1000 == 0):
		print(i/1000)
	for p in range(LIMIT):
		proj = np.dot(l,evcts[:, evls[p][0]])
		l = l-(proj*evcts[:, evls[p][0]])
		errs[p] += np.linalg.norm(l)

errs = errs/float(NUM)	
plt.plot(errs[:LIMIT])
plt.show()	