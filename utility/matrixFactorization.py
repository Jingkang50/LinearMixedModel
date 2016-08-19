__author__ = 'Haohan Wang'

import numpy as np
from dataLoader import EEGLoading, GenLoading, RanLoading

def Factorization(K):
    S, U = np.linalg.eigh(K)
    return S, U

def runEEG():
    X, y, Z0, Z1 = EEGLoading()

    print 'EEG Z0'
    K0 = np.dot(Z0, Z0.T)
    K0 = K0[:10000, :10000]
    S0, U0 = Factorization(K0)

    np.savetxt('../Data/EEGdata/K0.csv', K0, delimiter=',')
    np.savetxt('../Data/EEGdata/S0.csv', S0, delimiter=',')
    np.savetxt('../Data/EEGdata/U0.csv', U0, delimiter=',')

    print 'EEG Z1'
    K1 = np.dot(Z1, Z1.T)
    K1 = K1[:10000, :10000]
    S1, U1 = Factorization(K1)

    np.savetxt('../Data/EEGdata/K1.csv', K1, delimiter=',')
    np.savetxt('../Data/EEGdata/S1.csv', S1, delimiter=',')
    np.savetxt('../Data/EEGdata/U1.csv', U1, delimiter=',')

    print 'EEG Z2'
    Z2 = np.append(Z0, Z1, 1)

    K2 = np.dot(Z2, Z2.T)
    K2 = K2[:10000, :10000]
    S2, U2 = Factorization(K2)

    np.savetxt('../Data/EEGdata/K2.csv', K2, delimiter=',')
    np.savetxt('../Data/EEGdata/S2.csv', S2, delimiter=',')
    np.savetxt('../Data/EEGdata/U2.csv', U2, delimiter=',')

def runGenome():
    X, Y, Z0, Z1 = GenLoading(False)
    #
    # print 'AT Z0'
    # K0 = np.dot(Z0, Z0.T)
    # S0, U0 = Factorization(K0)
    #
    # np.savetxt('../Data/ATdata/K0.csv', K0, delimiter=',')
    # np.savetxt('../Data/ATdata/S0.csv', S0, delimiter=',')
    # np.savetxt('../Data/ATdata/U0.csv', U0, delimiter=',')
    #
    # print 'AT Z1'
    # K1 = np.dot(Z1, Z1.T)
    # S1, U1 = Factorization(K1)
    #
    # np.savetxt('../Data/ATdata/K1.csv', K1, delimiter=',')
    # np.savetxt('../Data/ATdata/S1.csv', S1, delimiter=',')
    # np.savetxt('../Data/ATdata/U1.csv', U1, delimiter=',')
    #
    print 'AT Z2'
    Z2 = np.append(Z0, Z1, 1)

    K2 = np.dot(Z2, Z2.T)
    S2, U2 = Factorization(K2)

    np.savetxt('../Data/ATdata/K2.csv', K2, delimiter=',')
    np.savetxt('../Data/ATdata/S2.csv', S2, delimiter=',')
    np.savetxt('../Data/ATdata/U2.csv', U2, delimiter=',')

    print 'AT Z4'

    Z4 = Z0 + Z1
    K4 = np.dot(Z4, Z4.T)
    S4, U4 = Factorization(K4)

    np.savetxt('../Data/ATdata/K4.csv', K4, delimiter=',')
    np.savetxt('../Data/ATdata/S4.csv', S4, delimiter=',')
    np.savetxt('../Data/ATdata/U4.csv', U4, delimiter=',')

def runRan():
    X, y, Z0, Z1 = RanLoading()

    print 'Random data Z0'
    K0 = np.dot(Z0, Z0.T)
    S0, U0 = Factorization(K0)

    np.savetxt('../Data/RandomData/K0.csv', K0, delimiter=',')
    np.savetxt('../Data/RandomData/S0.csv', S0, delimiter=',')
    np.savetxt('../Data/RandomData/U0.csv', U0, delimiter=',')

    print 'Random data Z1'
    K1 = np.dot(Z1, Z1.T)
    S1, U1 = Factorization(K1)

    np.savetxt('../Data/RandomData/K1.csv', K1, delimiter=',')
    np.savetxt('../Data/RandomData/S1.csv', S1, delimiter=',')
    np.savetxt('../Data/RandomData/U1.csv', U1, delimiter=',')

    print 'Random data Z2'
    Z2 = np.append(Z0, Z1, 1)

    K2 = np.dot(Z2, Z2.T)
    S2, U2 = Factorization(K2)

    np.savetxt('../Data/RandomData/K2.csv', K2, delimiter=',')
    np.savetxt('../Data/RandomData/S2.csv', S2, delimiter=',')
    np.savetxt('../Data/RandomData/U2.csv', U2, delimiter=',')

    Z4 = Z0 + Z1
    K4 = np.dot(Z4, Z4.T)
    S4, U4 = Factorization(K4)

    np.savetxt('../Data/RandomData/K4.csv', K4, delimiter=',')
    np.savetxt('../Data/RandomData/S4.csv', S4, delimiter=',')
    np.savetxt('../Data/RandomData/U4.csv', U4, delimiter=',')


if __name__ == '__main__':
    # runEEG()
    # runGenome()
    runRan()
