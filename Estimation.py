import numpy as np

# Step 1: Load in data once generated.

npzfile = np.load('./Data/smalldata/data.npz')
X = npzfile['arr_1']
Y = npzfile['arr_3']
G = npzfile['arr_4']
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
res = lmm_lasso.train(X,K,Y,0.5)
beta = res["weights"]
print len(beta)
np.savetxt("./Data/smalldata/lasso_B.csv", beta, '%5.2f',delimiter=",")
np.savez("./Data/smalldata/ML",B_reml,B,beta)

# num = 10
# dim = 10
# try_x = X[0:num][0:dim]
# try_y = Y[0:num]
# try_g = G[0:num]
# try_z = np.zeros([np.shape(try_x)[0],np.amax(try_g)+1])
# for i in range(1,np.shape(try_g)[0]):
#     try_z[i-1][try_g[i]] = 1
# try_k = lmm.calculateKinship(try_z)
# try_L = lmm.LMM(try_y,try_k)

