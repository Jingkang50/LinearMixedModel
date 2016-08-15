__author__ = 'Haohan Wang'

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import numpy as np
from dataLoader import EEGLoading, GenLoading, GenLoadingCausal

def limitPrediction(l, num):
    s = sorted(l)
    t = s[-num]
    r = []
    for v in l:
        if v > t:
            r.append(v-t)
        else:
            r.append(0)
    return r

def getPositions(l):
    text = [line.strip() for line in open('../Data/ATdata/athaliana.snps.chromPositionInfo.txt')][1]
    # print 'This position information is only for AT'
    pos = text.split()[:l]
    pos = [int(k) for k in pos]
    return pos

def getNearbyIndex(k, positions, nearby):
    k = int(k)
    mini = k
    maxi = k
    pos = positions[k]
    while mini>=1 and abs(positions[mini] - pos) < nearby:
        mini -=1
    l = len(positions)
    while maxi<l-2 and abs(positions[maxi] - pos) < nearby:
        maxi += 1
    return mini, maxi

def gwas_roc(weights, causal_snps, positions=None, nearby=1000):
    weights = limitPrediction(weights, 1000)

    score = np.array(weights)
    label = np.zeros(len(weights))

    if positions is None:
        positions = getPositions(len(score))
    for k in causal_snps:
        mini, maxi = getNearbyIndex(k, positions, nearby)
        i = np.argmax(score[mini:maxi])
        label[mini+i] = 1
    fpr, tpr, t = roc_curve(label, score)

    return fpr, tpr

def precision_recall(beta_true, beta_pred):
    b_true = beta_true!=0
    b_pred = beta_pred!=0
    p, r, f, s = prfs(b_true, b_pred, pos_label=1)
    return p, r

def Roc_curve(beta_true, beta_pred_list, labels):
    from matplotlib import pyplot as plt

    for i in range(len(beta_pred_list)):
        fpr, tpr = gwas_roc(beta_pred_list[i], beta_true, nearby=1000)
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
        # m = np.median(y_pred)
        # assert m<1
        # y_pred[y_pred>m] = 1
        # y_pred[y_pred<m] = 0
        a = accuracy_score(y_true, y_pred.astype(int))
        r.append(a)
        # plt.scatter(x, y_pred)
        # plt.show()
    # plt.scatter(x, y_true)
    # plt.show()
    return r

def evaluationGen():
    B = GenLoadingCausal()
    for i in range(4):
        print '-------------------'
        print 'Confound ', i
        beta_pred_list = np.loadtxt('../results/genomeResult_con_'+str(i+1)+'.csv', delimiter=',')
        Roc_curve(B, beta_pred_list, ['linear', 'L1', 'L2'])
        print '-------------------'

def evaluationEEG():
    X, Y, Z0, Z1 = EEGLoading()
    for k in range(2):
        print '============'
        print 'Label', k
        y_true = Y[10000:, k]
        for i in range(4):
            print '------------'
            print 'Confound', i
            y_pred_list = np.loadtxt('../results/EEGResult_label_'+str(k+1)+'_con_'+str(i+1)+'.csv', delimiter=',')
            print accuracy(y_true, y_pred_list)
            print '------------'
        print '============'


if __name__ == '__main__':
    evaluationGen()
    evaluationEEG()


