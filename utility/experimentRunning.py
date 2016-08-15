__author__ = 'Haohan Wang'

import sys

sys.path.append('../')

from dataLoader import GenLoading, EEGLoading, EEGLoading_KSU, GenLoadingKSU
from LMM.lmm_lasso import *
from evaluation import precision_recall

methods = ['linear', 'lasso', 'ridge']


def runEEG(numintervals=100, ldeltamin=-5, ldeltamax=5):
    X, y, Z0, Z1 = EEGLoading()

    for l in range(2):
        print 'EEG label', l
        Y = y[:, l]
        for i in range(3):
            print 'con ', i
            K, U, S = EEGLoading_KSU(i)

            Xtr = X[:10000, :]
            Ytr = Y[:10000]
            Xte = X[10000:, :]

            print 'linear'
            w_linear, alp, l_linear, clf_linear = train(Xtr, K, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin,
                                                        ldeltamax=ldeltamax, method='linear', selectK=False, regression=False, S=S, U=U)
            print 'lasso'
            w_lasso, alp, l_lasso, clf_lasso = train(Xtr, K, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin,
                                                     ldeltamax=ldeltamax, method='lasso', selectK=False, regression=False, S=S, U=U)
            print 'ridge'
            w_rd, alp, l_rd, clf_rd = train(Xtr, K, Ytr, mu=0, numintervals=numintervals, ldeltamin=ldeltamin,
                                            ldeltamax=ldeltamax, method='ridge', selectK=False, regression=False, S=S, U=U)

            y_pred_linear = predict(Xte, None, l_linear, clf_linear)
            y_pred_lasso = predict(Xte, None, l_lasso, clf_lasso)
            y_pred_rd = predict(Xte, None, l_rd, clf_rd)

            m = []
            m.append(y_pred_linear)
            m.append(y_pred_lasso)
            m.append(y_pred_rd)
            m = np.array(m)
            np.savetxt('../results/EEGResult_label_'+str(l+1)+'_con_'+str(i+1)+'.csv', m, delimiter=',')


def runGenome(numintervals=100, ldeltamin=-5, ldeltamax=5):
    X, Y, Z1, Z2, B = GenLoading(True)
    for i in range(3):
        print 'Genome', i
        K, U, S = GenLoadingKSU(i)
        w_linear, alp, l_linear, clf_linear = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin,
                                                    ldeltamax=ldeltamax, method='linear', selectK=True, regression=True, S=S, U=U)
        w_lasso, alp, l_lasso, clf_lasso = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin,
                                                 ldeltamax=ldeltamax, method='lasso', selectK=True, regression=True, S=S, U=U)
        w_rd, alp, l_rd, clf_rd = train(X, K, Y, mu=0, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax,
                                        method='ridge', selectK=True, regression=True, S=S, U=U)
        m = []
        m.append(w_linear)
        m.append(w_lasso)
        m.append(w_rd)
        m = np.array(m)
        np.savetxt('../results/genomeResult_con_'+str(i+1)+'.csv', m, delimiter=',')


if __name__ == '__main__':
    runGenome(1000, -10, 10)
    runEEG(1000, -10, 10)
