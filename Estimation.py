import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load in data once generated.

npzfile = np.load('./Data/smalldata/data.npz')
X = npzfile['arr_1']
beta_true = npzfile['arr_2']
Y = npzfile['arr_3']
G = npzfile['arr_4']

beta_true = beta_true.reshape((-1,))
print np.shape(X)
print np.shape(Y)
print np.shape(G)
# B = npzfile['arr_2'] need to derive

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
B_reml = lmm.GWAS(Y,X,K)
end = time.time()
sys.stderr.write("Total time for 100 SNPs: %0.3f\n" % (end- begin))
# print B_reml
np.savetxt("./Data/smalldata/REML_B.csv", B_reml, '%5.2f',delimiter=",")
B = lmm.GWAS(Y,X,K,REML=False)
np.savetxt("./Data/smalldata/ML_B.csv", B, '%5.2f',delimiter=",")

import lmm_lasso
# hyperparameters for lasso
mu = 1
min = -5
max = 5
numintv=100
rho= 1
alpha=0.9

res = lmm_lasso.train(X,K,Y,mu=mu,numintervals=numintv,ldeltamin=min,ldeltamax=max,rho=rho,alpha=alpha)
beta = res["weights"]
print len(beta)
np.savetxt("./Data/smalldata/lasso_B.csv", beta, '%5.2f',delimiter=",")
np.savez("./Data/smalldata/ML",B_reml,B,beta)



beta_true = beta_true.reshape((-1,))
beta_lasso = beta.reshape((-1,))
beta_ml = np.asarray(B).reshape((-1,))
beta_reml = np.asarray(B_reml).reshape((-1,))

# function for calculating ROC
def calROC(threshod , method):
    beta_true_r = np.where(beta_true > 0, 1 , 0)
    ones = float(np.count_nonzero(beta_true_r))
    beta_ml_r = np.where(beta_ml > threshod, 1 , 0)
    beta_reml_r = np.where(beta_reml > threshod, 1, 0)
    beta_lasso_r = np.where(beta_lasso > threshod , 1 ,0)
    if method == 'ml':
        mat = beta_true_r - beta_ml_r
        col = 'b'
    if method == 'reml':
        mat = beta_true_r - beta_reml_r
        col = 'g'
    if method == 'lasso':
        mat = beta_true_r - beta_lasso_r
        col = 'r'
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
    return TPR, FPR, col

def drawROC(start , end , num, method):
    TPR_set = []
    FPR_set = []
    for i in np.linspace(start , end , num):
        TPR, FPR, col = calROC(i, method)
        TPR_set.append(TPR)
        FPR_set.append(FPR)
    plt.plot(FPR_set, TPR_set, col ,marker='o',markersize = 10)
    # print TPR_set
    # print FPR_set
drawROC(-10, 10 ,100, 'ml')
drawROC(-10, 10, 100, 'reml')
drawROC(-10, 10 , 100, 'lasso')
plt.show()