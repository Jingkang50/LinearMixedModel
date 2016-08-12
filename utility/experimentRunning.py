__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

from dataLoader import GenLoading, EEGLoading
from LMM.lmm_lasso import *
from evaluation import precision_recall

methods = ['linear', 'lasso', 'ridge']

def runEEG(numintervals=100, ldeltamin=-5, ldeltamax=5):
    X, y, Z0, Z1 = EEGLoading()
    Y = y[:, 0]
    K = np.dot(Z0, Z0.T)

    Xtr = X[:10000, :]
    Ytr = Y[:10000]
    Ktr = K[:10000, :10000]
    Xte = X[10000:, :]
    Kte = K[10000:, :10000]

    w_linear, alp, l_linear = train(Xtr, Ktr, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='linear', selectK=False)
    w_lasso, alp, l_lasso = train(Xtr, Ktr, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='lasso', selectK=False)
    w_rd, alp, l_rd = train(Xtr, Ktr, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='ridge', selectK=False)

    y_pred_linear = predict(Ytr, Xtr, Xte, Ktr, Kte, l_linear, w_linear)
    y_pred_lasso = predict(Ytr, Xtr, Xte, Ktr, Kte, l_lasso, w_lasso)
    y_pred_rd = predict(Ytr, Xtr, Xte, Ktr, Kte, l_rd, w_rd)

    m = []
    m.append(y_pred_linear)
    m.append(y_pred_lasso)
    m.append(y_pred_rd)
    m = np.array(m)
    np.savetxt('EEGResult.csv', m, delimiter=',')

def runGenome(numintervals=100, ldeltamin=-5, ldeltamax=5):
    X, Y, Z, B = GenLoading(True)
    K = np.dot(Z, Z.T)
    w_linear, alp, l_linear = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='linear', selectK=True)
    w_lasso, alp, l_lasso = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='lasso', selectK=True)
    w_rd, alp, l_rd = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, method='ridge', selectK=True)
    m = []
    m.append(w_linear)
    m.append(w_lasso)
    m.append(w_rd)
    m = np.array(m)
    np.savetxt('genomeResult.csv', m, delimiter=',')


if __name__ == '__main__':
    runGenome(1000, -10, 10)
    runEEG(1000, -10, 10)
