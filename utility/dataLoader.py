__author__ = 'Haohan Wang'

import numpy as np
from numpy import genfromtxt

def EEGLoading():
	file = genfromtxt('../Data/EEGdata.csv', delimiter=',')
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

def GenLoading(returnB = False):
	Xdata = genfromtxt('../Data/smalldata/X.csv')
	Ydata = genfromtxt('../Data/smalldata/Y.csv')
	Gdata = genfromtxt('../Data/smalldata/G.csv')
	Bdata = genfromtxt('../Data/smalldata/B.csv')
	X = np.asarray(Xdata)
	Y = np.asarray(Ydata)
	G = np.asarray(Gdata)
	B = np.asarray(Bdata)
	Z = np.zeros([np.shape(X)[0], np.amax(G) + 1])
	for i in range(1, np.shape(G)[0]):
		Z[i - 1][G[i]] = 1
	if returnB == False:	return X, Y, Z
	else: return X, Y, Z, B

# X, Y, Z, B = GenLoading(returnB=True)
# print X.shape
# print Y.shape
# print Z.shape
# print B.shape
