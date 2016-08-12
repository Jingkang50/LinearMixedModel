__author__ = 'Haohan Wang'

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import roc_curve
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


if __name__ == '__main__':
    from dataLoader import EEGLoading, GenLoading
    X, Y, Z, B = GenLoading(True)
    beta_pred_list = np.loadtxt('../results/genomeResult.csv', delimiter=',')
    precison_recall_curve(B, beta_pred_list, ['linear', 'L1', 'L2'])