__author__ = 'Haohan Wang'

from lmm_lasso import *


# https://github.com/yangta1995/LinearMixedModel


def trainMulti(X, KList, y, mu, method='linear', numintervals=100, ldeltamin=-5, ldeltamax=5, selectK=True, SK=1000,
               regression=True, SList=None, UList=None, REML=False):
    [n_s, n_f] = X.shape
    # assert X.shape[0] == y.shape[0], 'dimensions do not match'
    # assert K.shape[0] == K.shape[1], 'dimensions do not match'
    # assert K.shape[0] == X.shape[0], 'dimensions do not match'
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))

    # train null model
    SList, UList, ldelta0 = train_nullmodel_multi(y, KList, numintervals, ldeltamin, ldeltamax, SList=SList,
                                                  UList=UList, REML=REML)

    # train lasso on residuals
    SUX = X
    SUy = y

    for i in range(len(KList)):
        U = UList[i]
        S = SList[i]

        delta0 = scipy.exp(ldelta0[i])
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, SUX)
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
        if regression:
            regList = []
            for i in range(20):
                regList.append(10 ** (i-20))
        else:
            regList = []
            for i in range(0, 10):
                regList.append(10 ** (i-15))
        alpha, ss = cv_train(SUX, SUy, regList, method, selectK, K=SK, regression=regression)
        w, clf = train_linear(SUX, SUy, alpha, method, regression)

    return w, alpha, ldelta0, clf


def train_nullmodel_multi(y, KList, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, SList=None, UList=None, REML=False):
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
    if SList is None or UList is None:
        SList = []
        UList = []
        for i in range(len(KList)):
            S, U = linalg.eigh(KList[i])
            SList.append(S)
            UList.append(U)

    lg_list = []

    for i in range(len(KList)):
        U = UList[i]
        S = SList[i]

        Uy = scipy.dot(U.T, y)

        # grid search
        nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        nllmin = scipy.inf
        for i in scipy.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S, REML)

        # find minimum
        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        # more accurate search around the minimum of the grid search

        for i in scipy.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S, REML),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        lg_list.append(ldeltaopt_glob)

    return SList, UList, lg_list
