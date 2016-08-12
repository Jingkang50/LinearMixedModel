__author__ = 'Haohan Wang'

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import numpy as np

def precision_recall(beta_true, beta_pred):
    b_true = beta_true!=0
    b_pred = beta_pred!=0
    p, r, f, s = prfs(b_true, b_pred, pos_label=1)
    return p, r

def precison_recall_curve(beta_true, beta_pred_list, labels):
    from matplotlib import pyplot as plt

    beta_pred_list = clean(beta_pred_list)
    b_true = beta_true!=0
    for i in range(len(beta_pred_list)):
        fpr, tpr, t = roc_curve(b_true, beta_pred_list[i], pos_label=1)
        plt.plot(fpr, tpr, label=labels[i])
    plt.legend()
    plt.show()

def clean(ll, k=100):
    r = []
    for l in ll:
        m = sorted(l)
        t = m[-k]
        l[l<t] = 0
        r.append(l)
    return r

def accuracy(y_true, y_pred_l):
    from matplotlib import pyplot as plt
    r = []
    x = xrange(len(y_true))
    for y_pred in y_pred_l:
        m = np.median(y_pred)
        assert m<1
        y_pred[y_pred>m] = 1
        y_pred[y_pred<m] = 0
        a = accuracy_score(y_true, y_pred.astype(int))
        r.append(a)
        # plt.scatter(x, y_pred)
        # plt.show()
    # plt.scatter(x, y_true)
    # plt.show()
    return r

if __name__ == '__main__':
    from dataLoader import EEGLoading, GenLoading
    X, Y, Z, B = GenLoading(True)
    beta_pred_list = np.loadtxt('../results/genomeResult.csv', delimiter=',')
    precison_recall_curve(B, beta_pred_list, ['linear', 'L1', 'L2'])

    X, Y, Z0, Z1 = EEGLoading()
    y_pred_list = np.loadtxt('../results/EEGResult.csv', delimiter=',')
    # print Y.shape
    # print y_pred_list.shape
    print accuracy(Y[10000:, 0], y_pred_list)




