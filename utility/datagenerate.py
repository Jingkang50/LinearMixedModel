import scipy
import pandas as pd
import numpy as np

reader = pd.read_csv('../Data/ATdata/athaliana.snps.all.csv',dtype={'a': np.int8}, header=None, iterator=True)
try:
    df = reader.get_chunk(100000000)
except StopIteration:
    print "Iteration is stopped."
    
# config
clusterNum = 5
featureNum = 500
pe = 0.3
ph = 0.3


snps = np.asarray(df, dtype=float)[:-2]
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
y = (1-pe)*ypheno + pe*error
causal = np.array(zip(idx, w))
np.savetxt('../Data/ATdata/snps.i.pheno.csv', y, '%5.2f', delimiter=',')
np.savetxt('../Data/ATdata/snps.i.pheno.causal.csv', causal, '%5.2f', delimiter=',')

# population
# categories = [int(line.strip()) for line in open('../data/athaliana.snps.categories.txt')][:m]
# c = len(set(categories))
# group = np.random.normal(0, 1, c)
# ygroup = np.array([group[i] for i in categories])

# we consider use snps as group information
cl = KMeans(n_clusters=clusterNum)
y = cl.fit_predict(snps)
c = cl.cluster_centers_

v = []
for i in range(len(y)):
    v.append(np.sum(c[y[i],:]))
v = np.array(v)
ygroup = (v-v.mean())/v.std()
np.savetxt('../Data/ATdata/snps.n.group.csv', y, '%d',delimiter=',')

y = (1-pe)*(ph*ypheno + (1-ph)*ygroup) + pe*error
causal = np.array(zip(idx, w))
np.savetxt('../Data/ATdata/snps.n.pheno.csv', y, '%5.2f',delimiter=',')
np.savetxt('../Data/ATdata/snps.n.pheno.causal.csv', causal, '%5.2f', delimiter=',')
