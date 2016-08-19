from sklearn.cluster import KMeans
import numpy as np
import scipy
import scipy.sparse as sparse
import math

# Size configuration
n = 1000	         # Case Number
p = 10000	     # Feature
dense = 0.1      # density

# config
clusterNum1 = 5
clusterNum2 = 5
pe = 0.2
pg1 = 0.4
pg2 = 0.4

np.set_printoptions(precision=4,
                       threshold=10000,
                       linewidth=150)

# generate a random X matrix
X = sparse.random(n, p, density=0.5)    # X is a n x p matrix with the value from 0 to
X = X.A
X = np.around(X, decimals= 3)

np.savetxt('../Data/RandomData/random.genotype.csv', X, '%5.3f',delimiter=',')


[m, n] = X.shape
featureNum = p * dense
idx = scipy.random.randint(0,n,featureNum).astype(int)
idx = sorted(idx)
w = 1*np.random.normal(0, 1, size=featureNum)
ypheno = scipy.dot(X[:,idx],w)
ypheno = (ypheno-ypheno.mean())/ypheno.std()
ypheno = ypheno.reshape(ypheno.shape[0])
error = np.random.normal(0, 1, m)

# Clustering
halflength = math.floor(p / 2)
cl1 = KMeans(n_clusters=clusterNum1)
cl2 = KMeans(n_clusters=clusterNum2)
g_1 = cl1.fit_predict(X[:,0:halflength])
g_2 = cl2.fit_predict(X[:,halflength:-1])
c1 = cl1.cluster_centers_
c2 = cl2.cluster_centers_

v = []
for i in range(len(g_1)):
    v.append(np.sum(c1[g_1[i],:]))
v = np.array(v)
ygroup_1 = (v-v.mean())/v.std()
np.savetxt('../Data/RandomData/random.group1.csv', g_1, '%d',delimiter=',')

v = []
for i in range(len(g_2)):
    v.append(np.sum(c2[g_2[i],:]))
v = np.array(v)
ygroup_2 = (v-v.mean())/v.std()
np.savetxt('../Data/RandomData/random.group2.csv', g_2, '%d',delimiter=',')

y = (1-pe)*( pg1 * ygroup_1 + pg2 * ygroup_2 + (1-pg1-pg2)*ypheno) + pe*error
causal = np.array(zip(idx, w))
np.savetxt('../Data/RandomData/random.pheno.csv', y, '%5.2f',delimiter=',')
np.savetxt('../Data/RandomData/random.causal.csv', causal, '%5.2f', delimiter=',')


