from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from collections import Counter

# Size configuration
n = 100	    # Case Number
p = 10000	# Feature
k = 4       # Number of Group
h = 0.2     # Impact of Group
dense = 0.05 # density

config = np.array([n,p,k,h])
np.set_printoptions(precision=4,
                       threshold=10000,
                       linewidth=150)

# generate a random X matrix
X = np.random.random((n,p))     # X is a n x p matrix with the value from 0 to
X = np.around(X, decimals=2)
B = sparse.random(p, 1, density=dense)    # B is random sparse vector
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
print 'the outcome of clustering'
print Counter(G)
# Comprise to Y
Y = Y_1 * [[1-h]] + Y_2 * [[h]]
Y = np.around(Y, decimals=2)

print Y.shape

# Visualization
colors = ['g', 'r', 'b', 'y','p']
for i in range(n):
    gid = G[i]
    plt.plot(Y[i], 'o', markerfacecolor=colors[gid], marker='.',markersize = 10)

# for i in range(k):
#     cluster_center = clf.cluster_centers_[i]
#     plt.plot(cluster_center,'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=6)

plt.show()

print "Start save txt"
np.savez("./Data/smalldata/data", config, X, B, Y, G)
np.savetxt("./Data/smalldata/Y.csv", Y, '%5.2f', delimiter=",")
np.savetxt("./Data/smalldata/X.csv", X, '%5.2f', delimiter=",")
np.savetxt("./Data/smalldata/G.csv", G, '%5.2f', delimiter=",")
np.savetxt("./Data/smalldata/B.csv", B, '%5.2f', delimiter=",")
print "File saving done"


