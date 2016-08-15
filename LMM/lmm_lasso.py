import scipy
import scipy.linalg as linalg
import scipy.optimize as opt
# import pdb
# import matplotlib.pylab as plt
import time
import numpy as np

#https://github.com/yangta1995/LinearMixedModel

def stability_selection(X, K, y, mu, n_reps, f_subset, **kwargs):
    """
    run stability selection2

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty

    n_reps:   number of repetitions
    f_subset: fraction of datasets that is used for creating one bootstrap

    output:
    selection frequency for all Snps: n_f x 1
    """
    time_start = time.time()
    [n_s, n_f] = X.shape
    n_subsample = scipy.ceil(f_subset * n_s)
    freq = scipy.zeros(n_f)

    for i in range(n_reps):
        print 'Iteration %d' % i
        idx = scipy.random.permutation(n_s)[:n_subsample]
        res = train(X[idx], K[idx][:, idx], y[idx], mu, **kwargs)
        snp_idx = (res['weights'] != 0).flatten()
        freq[snp_idx] += 1.

    freq /= n_reps
    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs' % (time_diff)
    return freq


def train(X, K, y, mu, method='linear', numintervals=100, ldeltamin=-5, ldeltamax=5, selectK=True, SK=1000, regression=True, S=None, U=None):

    [n_s, n_f] = X.shape
    assert X.shape[0] == y.shape[0], 'dimensions do not match'
    assert K.shape[0] == K.shape[1], 'dimensions do not match'
    assert K.shape[0] == X.shape[0], 'dimensions do not match'
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))

    # train null model
    S, U, ldelta0 = train_nullmodel(y, K, numintervals, ldeltamin, ldeltamax, S=S, U=U)

    # train lasso on residuals
    delta0 = scipy.exp(ldelta0)
    Sdi = 1. / (S + delta0)
    Sdi_sqrt = scipy.sqrt(Sdi)
    SUX = scipy.dot(U.T, X)
    SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
    if regression:
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
    else:
        SUy = y.astype(int).reshape(y.shape[0])

    print SUy.shape

    if method == 'linear':
        w, clf = train_linear(SUX, SUy, mu, method, regression)
        alpha = 0
    else:
        regList = []
        for i in range(10):
            regList.append(10 ** (i-5))
        alpha, ss = cv_train(SUX, SUy, regList, method, selectK, K=SK, regression=regression)
        w, clf = train_linear(SUX, SUy, alpha, method, regression)

    return w, alpha, ldelta0, clf


def predict_old(y_t, X_t, X_v, K_tt, K_vt, ldelta, w):
    """
    predict the phenotype

    Input:
    y_t: phenotype: n_train x 1
    X_t: Snp matrix: n_train x n_f
    X_v: Snp matrix: n_val x n_f
    K_tt: kinship matrix: n_train x n_train
    K_vt: kinship matrix: n_val  x n_train
    ldelta: kernel parameter
    w: lasso weights: n_f x 1

    Output:
    y_v: predicted phenotype: n_val x 1
    """
    print 'predict LMM-Lasso'

    assert y_t.shape[0] == X_t.shape[0], 'dimensions do not match'
    assert y_t.shape[0] == K_tt.shape[0], 'dimensions do not match'
    assert y_t.shape[0] == K_tt.shape[1], 'dimensions do not match'
    assert y_t.shape[0] == K_vt.shape[1], 'dimensions do not match'
    assert X_v.shape[0] == K_vt.shape[0], 'dimensions do not match'
    assert X_t.shape[1] == X_v.shape[1], 'dimensions do not match'
    assert X_t.shape[1] == w.shape[0], 'dimensions do not match'

    [n_train, n_f] = X_t.shape
    n_test = X_v.shape[0]

    if y_t.ndim == 1:
        y_t = scipy.reshape(y_t, (n_train, 1))
    if w.ndim == 1:
        w = scipy.reshape(w, (n_f, 1))

    delta = scipy.exp(ldelta)
    idx = w.nonzero()[0]

    if idx.shape[0] == 0:
        return scipy.dot(K_vt, linalg.solve(K_tt + delta * scipy.eye(n_train), y_t))

    y_v = scipy.dot(X_v[:, idx], w[idx]) + scipy.dot(K_vt, linalg.solve(K_tt + delta * scipy.eye(n_train),
                                                                        y_t - scipy.dot(X_t[:, idx], w[idx])))
    return y_v

def predict(X, Kvt, ldelta, clf):
    # S, U = linalg.eigh(Kvt)
    # [n_s, n_f] = X.shape
    # delta0 = scipy.exp(ldelta)
    # Sdi = 1. / (S + delta0)
    # Sdi_sqrt = scipy.sqrt(Sdi)
    # SUX = scipy.dot(U.T, X)
    # SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T

    y = clf.predict(X)
    return y

"""
helper functions
"""


def train_linear(X, y, mu=1e-4, method='linear', regression=True):
    if not regression:
        if method == 'linear':
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(X, y)
            w = lr.coef_
            return w.reshape((w.shape[1],)), lr
        elif method == 'lasso':
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=mu)
            lasso.fit(X, y)
            return lasso.coef_, lasso
        elif method == 'ridge':
            from sklearn.linear_model import RidgeClassifier
            rc = RidgeClassifier(alpha=mu)
            rc.fit(X, y)
            return rc.coef_[0], rc
    else:
        if method == 'linear':
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            w = lr.coef_
            return w.reshape((w.shape[1],)), lr
        elif method == 'lasso':
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=mu)
            lasso.fit(X, y)
            return lasso.coef_, lasso
        elif method == 'ridge':
            from sklearn.linear_model import Ridge
            rc = Ridge(alpha=mu)
            rc.fit(X, y)
            return rc.coef_[0], rc


def nLLeval(ldelta, Uy, S, REML=True):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    if REML:
        pass

    return nLL


def train_nullmodel(y, K, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, S=None, U=None):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    ldeltamin += scale
    ldeltamax += scale

    n_s = y.shape[0]

    # rotate data
    if S is None or U is None:
        S, U = linalg.eigh(K)

    Uy = scipy.dot(U.T, y)

    # grid search
    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    nllmin = scipy.inf
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

    # find minimum
    nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    # more accurate search around the minimum of the grid search

    for i in scipy.arange(numintervals - 1) + 1:
        if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
            ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                          (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                          full_output=True)
            if nllopt < nllmin:
                nllmin = nllopt
                ldeltaopt_glob = ldeltaopt

    return S, U, ldeltaopt_glob


def cv_train(X, Y, regList, method, selectK=False, K=1000, regression=True):
    ss = []
    if not selectK:
        from sklearn import cross_validation
        b = np.inf
        breg = 0
        for reg in regList:
            if method == 'lasso':
                from sklearn.linear_model import Lasso
                clf = Lasso(alpha=reg)
            elif method == 'ridge':
                if regression:
                    from sklearn.linear_model import Ridge
                    clf = Ridge(alpha=reg)
                else:
                    from sklearn.linear_model import RidgeClassifier
                    clf = RidgeClassifier(alpha=reg)
            else:
                clf = None
            if regression:
                scores = cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='mean_squared_error')
            else:
                scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
            s = np.mean(np.abs(scores))
            print reg, s
            ss.append(s)
            if s < b:
                b = s
                breg = reg
        return breg, ss
    else:
        b = np.inf
        breg = 0
        for reg in regList:
            w = train_linear(X, Y, reg, method, regression)
            k = len(np.where(w > 0.01 )[0])
            # s = np.abs(k-K)
            if k < K:
                s = np.inf
            else:
                s = np.abs(k - K)
            print reg, s
            ss.append(s)
            if s < b:
                b = s
                breg = reg
        return breg, ss

