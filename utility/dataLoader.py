__author__ = 'Haohan Wang'

import numpy as np
from numpy import genfromtxt

def dataLoading(filename):
	file = genfromtxt(filename, delimiter=',')
	data = np.asarray(file)
	X = data[: , 2 :13 ]
	Y = data[: , 13:15 ]
	G0 = data[: , 0]
	G1 = data[: , 1]
	Z0 = np.zeros([np.shape(X)[0], np.amax(G0) + 1])
	for i in range(1, np.shape(G0)[0]):
		Z0[i - 1][G0[i]] = 1
	Z1 = np.zeros([np.shape(X)[0], np.amax(G1) + 1])
	for i in range(1, np.shape(G1)[0]):
		Z1[i - 1][G1[i]] = 1
	return X, Y, Z0, Z1
<<<<<<< HEAD
dataLoading('../Data/EEGdata.csv')
=======
# dataLoading('../Data/EEGdata.csv')
>>>>>>> origin/master
