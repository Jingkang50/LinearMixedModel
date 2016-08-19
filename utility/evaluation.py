__author__ = 'Haohan Wang'

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import numpy as np
from dataLoader import EEGLoading, GenLoading, GenLoadingCausal, RanLoadingCausal

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

def gwas_roc(weights, causal_snps, positions=None, top=1000, nearby=1000):
    weights = limitPrediction(weights, top)
    score = np.array(weights)
    label = np.zeros(len(weights))
    if positions is None:
        positions = getPositions(len(score))
    for k in causal_snps:
        mini, maxi = getNearbyIndex(k, positions, nearby)
        i = np.argmax(score[mini:maxi])
        label[mini+i] = 1
    # fpr, tpr, t = roc_curve(label, score)
    p, r, t = prc(label, score)
    # return fpr, tpr
    return r, p

def precision_recall(beta_true, beta_pred):
    b_true = beta_true!=0
    b_pred = beta_pred!=0
    p, r, f, s = prfs(b_true, b_pred, pos_label=1)
    return p, r

def Roc_curve(beta_true, beta_pred_list, labels, top, nearby, method):
    from matplotlib import pyplot as plt
    fpr_list = []
    tpr_list = []
    fig = plt.figure()
    for i in range(len(beta_pred_list)):
        fpr, tpr = gwas_roc(beta_pred_list[i], beta_true, top=top, nearby=nearby)
        plt.ylim(0, 1.05)
        plt.plot(fpr, tpr, label=labels[i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    plt.legend()
    # plt.show()
    fig.savefig('../pic/Gen_ROC_'+str(top)+'_'+str(nearby)+'_'+str(method)+'.png' , dpi=fig.dpi)
    plt.close()
    auc_list = []
    for j in range(len(fpr_list)):
        # print auc(fpr_list[j], tpr_list[j])
        auc_list.append(auc(fpr_list[j], tpr_list[j]))
    return auc_list

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

def pr_score(y_true, y_pred_l):
    from matplotlib import pyplot as plt
    result = []
    x = xrange(len(y_true))
    c = -1
    ls = ['linear', 'L1', 'L2']
    for y_pred in y_pred_l:
        c += 1
        m = f1_score(y_true, y_pred.astype(int))
        result.append(m)
        # plt.plot(p, r, label=ls[c])
    # plt.legend()
    # plt.show()
    return result

def evaluationGen(top, nearby):
    B = GenLoadingCausal()
    auc_list = []
    compare_list = []
    method = ['linear', 'L1', 'L2']
    for t in ['ML', 'REML']:
        for i in range(5):
            print '-------------------'
            print 'Confound ', i
            beta_pred_list = np.loadtxt('../results/genomeResult_'+t+'_con_'+str(i+1)+'.csv', delimiter=',')
            auc = Roc_curve(B, beta_pred_list, method, top, nearby,i)
            auc_arr = np.asarray(auc)
            print method[np.argmax(auc_arr)]
            auc_list.append(auc)
            compare_list.append(method[np.argmax(auc)])
            compare_list.append(auc_arr[0])
            compare_list.append(auc_arr[1])
            compare_list.append(auc_arr[2])
    # np.savetxt('../Data/GenEva.csv', auc_list, '%5.4f' ,delimiter=',')
    return compare_list
    # np.savetxt('../Data/GenComp.txt', np.asarray(compare_list), delimiter=',',fmt='%s')

def evaluationRan(top, nearby):
    B = RanLoadingCausal()
    auc_list = []
    compare_list = []
    method = ['linear', 'L1', 'L2']
    for t in ['ML', 'REML']:
        for i in range(5):
            print '-------------------'
            print 'Confound ', i
            beta_pred_list = np.loadtxt('../results/RandomDataResult_'+t+'_con_'+str(i+1)+'.csv', delimiter=',')
            auc = Roc_curve(B, beta_pred_list, method, top, nearby, i)
            auc_arr = np.asarray(auc)
            print method[np.argmax(auc_arr)]
            auc_list.append(auc)
            compare_list.append(method[np.argmax(auc)])
            compare_list.append(auc_arr[0])
            compare_list.append(auc_arr[1])
            compare_list.append(auc_arr[2])
    # np.savetxt('../Data/RanEva.csv', auc_list, '%5.4f' ,delimiter=',')
    return compare_list


def evaluationEEG():
    X, Y, Z0, Z1 = EEGLoading()
    method = ['linear', 'L1', 'L2']
    pre_list = []
    compare_list = []
    for t in ['ML', 'REML']:
        for k in range(2):
            print '============'
            print 'Label', k
            for i in range(5):
                print '------------'
                print 'Confound', i
                # y_pred_list = np.loadtxt('../results/EEGResult_'+t+'_label_'+str(k+1)+'_con_'+str(i+1)+'.csv', delimiter=',')
                y_pred_list = np.loadtxt('../results2/EEGResult_'+t+'_label_'+str(k+1)+'_con_'+str(i+1)+'.csv', delimiter=',')
                y_true = Y[-y_pred_list.shape[1]:, k]
                print pr_score(y_true, y_pred_list)
                print method[np.argmax(accuracy(y_true, y_pred_list))]
                print '------------'
                pre_list.append(accuracy(y_true, y_pred_list))
                compare_list.append(method[np.argmax(accuracy(y_true, y_pred_list))])
            print '============'
    # np.savetxt('../Data/EEGEva.csv', pre_list, '%5.4f', delimiter=',')
    # np.savetxt('../Data/EEGComp.txt', np.asarray(compare_list), delimiter=',', fmt='%s')


def evaluationGen_chosen():
    full_comp = []
    chosen = [line.strip().split('\t') for line in open('../results/filter.txt')]
    for a, b in chosen:
        i = int(a)
    j = int(b)
    full_comp.append([i, j] + evaluationGen(i, j))
    print i
    np.savetxt('../Data/GenCompPR_chosen.csv', np.asarray(full_comp), delimiter=',', fmt='%s')


if __name__ == '__main__':
    # full_comp = []
    # for i in range(150,153,20):
    #     full_comp.append([i]+evaluationRan(i,1))
    #     print i
    #     np.savetxt('../Data/RandomDataPR_'+str(i)+'.csv', np.asarray(full_comp), delimiter=',',fmt='%s')
    for i in range(15000, 40000, 5000):
        evaluationGen(1000, i)
    # evaluationEEG()


