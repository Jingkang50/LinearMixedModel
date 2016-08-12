__author__ = 'Haohan Wang'

from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np

def precision_recall(beta_true, beta_pred):
    b_true = beta_true!=0
    b_pred = beta_pred!=0
    p, r, f, s = prfs(b_true, b_pred, pos_label=1)
    return p, r
