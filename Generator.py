from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sparse

# Size configuration
n = 500     # Case Number
p = 10000   # Feature
k = 5       # Number of Group
h = 0.4     # Impact of Group
config = np.array([n,p,k,h])

# generate a random X matrix
X = np.random.random((n,p))     # X is a n x p matrix with the value from 0 to
X = np.around(X, decimals=2)
B = sparse.random(p, 1, density=0.1)    # B is random sparse vector
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

print "Start save txt"
np.savez("./Data/LMM0706/1", config, X, B, Y, G)
np.savetxt("./Data/LMM0706/Y.csv", Y, delimiter=",")
np.savetxt("./Data/LMM0706/X.csv", X, delimiter=",")
np.savetxt("./Data/LMM0706/G.csv", G, delimiter=",")
np.savetxt("./Data/LMM0706/B.csv", B, delimiter=",")
np.savetxt("./Data/LMM0706/config.csv", config, delimiter=",")
print "File saving done"













# Instruction
# >>> from tempfile import TemporaryFile
# >>> outfile = TemporaryFile()
# >>> x = np.arange(10)
# >>> y = np.sin(x)
# Using savez with *args, the arrays are saved with default names.
#
# >>>
# >>> np.savez(outfile, x, y)
# >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
# >>> npzfile = np.load(outfile)
# >>> npzfile.files
# ['arr_1', 'arr_0']
# >>> npzfile['arr_0']
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Using savez with **kwds, the arrays are saved with the keyword names.
#
# >>>
# >>> outfile = TemporaryFile()
# >>> np.savez(outfile, x=x, y=y)
# >>> outfile.seek(0)
# >>> npzfile = np.load(outfile)
# >>> npzfile.files
# ['y', 'x']
# >>> npzfile['x']
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
