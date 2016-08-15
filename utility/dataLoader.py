__author__ = 'Haohan Wang'

import numpy as np
from numpy import genfromtxt


def EEGLoading():
    file = np.loadtxt('../Data/EEGdata/EEGdata.csv', delimiter=',')
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

def EEGLoading_KSU(i):
    K = np.loadtxt('../Data/EEGdata/K'+str(i)+'.csv', delimiter=',')
    U = np.loadtxt('../Data/EEGdata/U'+str(i)+'.csv', delimiter=',')
    S = np.loadtxt('../Data/EEGdata/S'+str(i)+'.csv', delimiter=',')
    return K, U, S


def GenLoading(returnB=False):
    Xdata = np.loadtxt('../Data/ATdata/athaliana.snps.all.csv', delimiter=',')
    Ydata = np.loadtxt('../Data/ATdata/snps.n.pheno.csv', delimiter=',')
    G1data = np.loadtxt('../Data/ATdata/snps.n.group1.csv', delimiter=',')
    G2data = np.loadtxt('../Data/ATdata/snps.n.group2.csv', delimiter=',')
    Bdata = np.loadtxt('../Data/ATdata/snps.n.pheno.causal.csv', delimiter=',')
    X = np.asarray(Xdata)
    Y = np.asarray(Ydata)
    G1 = np.asarray(G1data).astype(int)
    G2 = np.asarray(G2data).astype(int)
    B = np.asarray(Bdata)
    Z1 = np.zeros([np.shape(X)[0], np.amax(G1) + 1])
    for i in range(1, np.shape(G1)[0]):
        Z1[i - 1][G1[i]] = 1
    Z2 = np.zeros([np.shape(X)[0], np.amax(G2) + 1])
    for i in range(1, np.shape(G2)[0]):
        Z2[i - 1][G2[i]] = 1
    if returnB == False:
        return X, Y, Z1, Z2
    else:
        return X, Y, Z1, Z2, B

def GenLoadingKSU(i):
    K = np.loadtxt('../Data/ATdata/K'+str(i)+'.csv', delimiter=',')
    U = np.loadtxt('../Data/ATdata/U'+str(i)+'.csv', delimiter=',')
    S = np.loadtxt('../Data/ATdata/S'+str(i)+'.csv', delimiter=',')
    return K, U, S


# dataLoading('../Data/EEGdata.csv')

if __name__ == '__main__':
    # X, Y, Z0, Z1 = EEGLoading()
    # print Z0.shape
    X, Y, Z1, Z2, B = GenLoading(True)
    print sum(B!=0)
    print Z2.shape
