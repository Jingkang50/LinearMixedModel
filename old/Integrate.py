from collections import Counter

import numpy as np
import scipy.sparse as sparse
from sklearn.cluster import KMeans

# Size configuration
n = 100	    # Case Number
p = 10000	    # Feature
k = 5       # Number of Group
h = 0.2     # Impact of Group
dense = 0.3 # density
datapath = './Data/mediumdata/'


# other setting:
config = np.array([n,p,k,h])
np.set_printoptions(precision=4,
                       threshold=10000,
                       linewidth=150)


# generate a random X matrix
X = np.random.random((n,p))     
# X is a n x p matrix with the value from 0 to
X = np.around(X, decimals=2)
B = sparse.random(p, 1, density=dense)    
# B is random sparse vector
B = B.A
B = np.around(B, decimals=2)
Y_1 = X.dot(B)
Y_1 = np.around(Y_1, decimals=2)
print X.shape
print B.shape
print Y_1.shape


# Clustering
clf = KMeans(n_clusters = k)
s = clf.fit(Y_1)

# Generate the group
Y_2 = np.zeros([n,1])
for i in range(0,n):
    Y_2[i] = clf.cluster_centers_[clf.labels_[i]]
Y_2 = np.around(Y_2, decimals=2)
print Y_2.shape

G = clf.labels_
# Comprise to Y
Y = Y_1 * [[1-h]] + Y_2 * [[h]]
Y = np.around(Y, decimals=2)

print Y.shape

G = clf.labels_
print 'the outcome of clustering:' + Counter(G)

print "Start save txt"
np.savez(datapath+"data", config, X, B, Y, G)
np.savetxt(datapath+"Y.csv", Y, '%5.2f', delimiter=",")
np.savetxt(datapath+"X.csv", X, '%5.2f', delimiter=",")
np.savetxt(datapath+"G.csv", G, '%5.2f', delimiter=",")
np.savetxt(datapath+"B.csv", B, '%5.2f', delimiter=",")
print "File saving done"

##########################################################
##   					Estimation                       #
##########################################################

# Formulate matrix Z
Z = np.zeros([np.shape(X)[0],np.amax(G)+1])
for i in range(1,np.shape(G)[0]):
    Z[i-1][G[i]] = 1
# print Z

import lmm
import time
import sys
# calculate the kinship (why?)
K = lmm.calculateKinship(Z)
# return beta, sigma
# ML solution
begin = time.time()
B_reml = lmm.GWAS(Y, X, K)
end = time.time()
sys.stderr.write("Total time for 100 SNPs: %0.3f\n" % (end- begin))
# print B_reml
np.savetxt(datapath + "REML_B.csv", B_reml, '%5.2f',delimiter=",")
B_ml = lmm.GWAS(Y, X, K, REML=False)
np.savetxt(datapath + "ML_B.csv", B_ml, '%5.2f',delimiter=",")

import lmm_lasso
res = lmm_lasso.train(X, K, Y, 0.5)
beta = res["weights"]
print len(beta)
np.savetxt(datapath + "lasso_B.csv", beta, '%5.2f',delimiter=",")
np.savez(datapath + "ML",B_reml,B_ml,beta)

#################################################################
#						draw ROC								#
#################################################################
beta_true = B.reshape((-1,))
beta_reml = np.asarray(B_reml)
beta_ml   = np.asarray(B_ml)
beta_lasso = np.asarray(beta)
beta_ml = beta_ml.reshape((-1,))
beta_lasso = beta_lasso.reshape((-1,))
beta_reml = beta_reml.reshape((-1,))
print np.shape(beta_ml)
print np.shape(beta_lasso)
print np.shape(beta_reml)
print np.shape(beta_true)


def calROC(threshod , method):
    beta_true_r = np.where(beta_true > 0, 1 , 0)
    ones = float(np.count_nonzero(beta_true_r))
    beta_ml_r = np.where(beta_ml > threshod, 1 , 0)
    beta_reml_r = np.where(beta_reml > threshod, 1, 0)
    beta_lasso_r = np.where(beta_lasso > threshod , 1 ,0)
    if method == 'ml':
        mat = beta_true_r - beta_ml_r
    if method == 'reml':
        mat = beta_true_r - beta_reml_r
    if method == 'lasso':
        mat = beta_true_r - beta_lasso_r
    count = np.bincount(mat+1)
    countii = np.nonzero(count)[0]
    outcome = zip(countii , count)
    if len(outcome) == 3:
        TPR = 1 - outcome[2][1] / ones
        FPR = outcome[0][1]/ (np.shape(beta_reml)[0] - ones)
    else:
        if outcome[0][0] == 0:
            TPR = 1
            FPR = outcome[0][1]/ (np.shape(beta_reml)[0] - ones)
        else:
            TPR = 0
            FPR = 0
    # print outcome
    return TPR, FPR

import matplotlib.pyplot as plt
def drawROC(start , end , num, method):
    TPR_set = []
    FPR_set = []
    for i in np.linspace(start , end , num):
        TPR, FPR = calROC(i, method)
        TPR_set.append(TPR)
        FPR_set.append(FPR)
    plt.plot(FPR_set, TPR_set)   
    # print TPR_set
    # print FPR_set
drawROC(-10 , 10 ,1000, 'ml')
drawROC(-10, 10, 1000, 'reml')
drawROC(-1, 2 , 1000, 'lasso')
plt.show()