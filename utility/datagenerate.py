import scipy
import pandas as pd
import numpy as np
import math

reader = pd.read_csv('../Data/ATdata/athaliana.snps.all.csv',dtype={'a': np.int8}, header=None, iterator=True)
try:
    df = reader.get_chunk(100000000)
except StopIteration:
    print "Iteration is stopped."
    
# config
clusterNum1 = 5
clusterNum2 = 5
featureNum = 500
pe = 0.3
pg1 = 0.3
pg2 = 0.3

snps = np.asarray(df, dtype=float)
snps.astype('int')

[m, n] = snps.shape
idx = scipy.random.randint(0,n,featureNum).astype(int)
idx = sorted(idx)
w = 1*np.random.normal(0, 1, size=featureNum)
ypheno = scipy.dot(snps[:,idx],w)
ypheno = (ypheno-ypheno.mean())/ypheno.std()
ypheno = ypheno.reshape(ypheno.shape[0])
error = np.random.normal(0, 1, m)

from sklearn.cluster import KMeans
# single
# y = (1-pe)*ypheno + pe*error
# causal = np.array(zip(idx, w))
# np.savetxt('../Data/ATdata/snps.i.pheno.csv', y, '%5.2f', delimiter=',')
# np.savetxt('../Data/ATdata/snps.i.pheno.causal.csv', causal, '%5.2f', delimiter=',')

# population
# categories = [int(line.strip()) for line in open('../data/athaliana.snps.categories.txt')][:m]
# c = len(set(categories))
# group = np.random.normal(0, 1, c)
# ygroup = np.array([group[i] for i in categories])

# we consider use snps as group information
halflength = math.floor(n / 2)
cl1 = KMeans(n_clusters=clusterNum1)
cl2 = KMeans(n_clusters=clusterNum2)
g_1 = cl1.fit_predict(snps[:,0:halflength])
g_2 = cl2.fit_predict(snps[:,halflength:-1])
c1 = cl1.cluster_centers_
c2 = cl2.cluster_centers_
v = []

for i in range(len(g_1)):
    v.append(np.sum(c1[g_1[i],:]))
v = np.array(v)
ygroup_1 = (v-v.mean())/v.std()
np.savetxt('../Data/ATdata/snps.n.group1.csv', g_1, '%d',delimiter=',')

v = []
for i in range(len(g_2)):
    v.append(np.sum(c2[g_2[i],:]))
v = np.array(v)
ygroup_2 = (v-v.mean())/v.std()
np.savetxt('../Data/ATdata/snps.n.group2.csv', g_2, '%d',delimiter=',')

y = (1-pe)*( pg1 * ygroup_1 + pg2 * ygroup_2 + (1-pg1-pg2)*ypheno) + pe*error
causal = np.array(zip(idx, w))
np.savetxt('../Data/ATdata/snps.n.pheno.csv', y, '%5.2f',delimiter=',')
np.savetxt('../Data/ATdata/snps.n.pheno.causal.csv', causal, '%5.2f', delimiter=',')
