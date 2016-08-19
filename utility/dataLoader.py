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

def EEGLoadingKSU(i):
    K = np.loadtxt('../Data/EEGdata/K'+str(i)+'.csv', delimiter=',')
    U = np.loadtxt('../Data/EEGdata/U'+str(i)+'.csv', delimiter=',')
    S = np.loadtxt('../Data/EEGdata/S'+str(i)+'.csv', delimiter=',')
    return K, U, S


def GenLoading(returnB=False):
    Xdata = np.loadtxt('../Data/ATdata/athaliana.snps.chrom1.csv', delimiter=',')
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

def GenLoadingCausal():
    Bdata = np.loadtxt('../Data/ATdata/snps.n.pheno.causal.csv', delimiter=',')
    B = np.asarray(Bdata)
    return B[:,0]
    # r = np.zeros((52172,))
    # for i in range(B.shape[0]):
    #     r[B[i,0]] = 1
    # return r

# dataLoading('../Data/EEGdata.csv')

def RanLoading(returnB=False):
    Xdata = np.loadtxt('../Data/RandomData/random.genotype.csv', delimiter=',')
    Ydata = np.loadtxt('../Data/RandomData/random.pheno.csv', delimiter=',')
    G1data = np.loadtxt('../Data/RandomData/random.group1.csv', delimiter=',')
    G2data = np.loadtxt('../Data/RandomData/random.group2.csv', delimiter=',')
    Bdata = np.loadtxt('../Data/RandomData/random.causal.csv', delimiter=',')
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

def RanLoadingKSU(i):
    K = np.loadtxt('../Data/RandomData/K'+str(i)+'.csv', delimiter=',')
    U = np.loadtxt('../Data/RandomData/U'+str(i)+'.csv', delimiter=',')
    S = np.loadtxt('../Data/RandomData/S'+str(i)+'.csv', delimiter=',')
    return K, U, S

def RanLoadingCausal():
    Bdata = np.loadtxt('../Data/RandomData/random.causal.csv', delimiter=',')
    B = np.asarray(Bdata)
    return B[:,0]


if __name__ == '__main__':
    # X, Y, Z0, Z1 = EEGLoading()
    # print Z0.shape
    X, Y, Z1, Z2, B = RanLoading(True)
    print sum(B!=0)
    print Z2.shape
