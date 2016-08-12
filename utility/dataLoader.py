__author__ = 'Haohan Wang'

import numpy as np
from numpy import genfromtxt


def EEGLoading():
    file = np.loadtxt('../Data/EEGdata.csv', delimiter=',')
    data = np.asarray(file)
    X = data[:, 2:13]
    Y = data[:, 13:15]
    G0 = data[:, 0].astype(int)
    G1 = data[:, 1].astype(int)
    Z0 = np.zeros([np.shape(X)[0], np.amax(G0) + 1])
    for i in range(1, np.shape(G0)[0]):
        Z0[i - 1][G0[i]] = 1
    Z1 = np.zeros([np.shape(X)[0], np.amax(G1) + 1])
    for i in range(1, np.shape(G1)[0]):
        Z1[i - 1][G1[i]] = 1
    return X, Y, Z0, Z1


def GenLoading(returnB=False):
    Xdata = np.loadtxt('../Data/mediumdata/X.csv', delimiter=',')
    Ydata = np.loadtxt('../Data/mediumdata/Y.csv', delimiter=',')
    Gdata = np.loadtxt('../Data/mediumdata/G.csv', delimiter=',')
    Bdata = np.loadtxt('../Data/mediumdata/B.csv', delimiter=',')
    X = np.asarray(Xdata)
    Y = np.asarray(Ydata)
    G = np.asarray(Gdata).astype(int)
    B = np.asarray(Bdata)
    Z = np.zeros([np.shape(X)[0], np.amax(G) + 1])
    for i in range(1, np.shape(G)[0]):
        Z[i - 1][G[i]] = 1
    if returnB == False:
        return X, Y, Z
    else:
        return X, Y, Z, B


# dataLoading('../Data/EEGdata.csv')

if __name__ == '__main__':
    X, Y, Z0, Z1 = EEGLoading()
    print Z0.shape
    X, Y, Z, B = GenLoading(True)
    print sum(B!=0)
