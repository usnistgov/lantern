from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


def sparse_design(X, order=3, interactions=None, addnew=False):
    I, J = np.where(X.numpy())

    # need to add more
    II = I.tolist()
    JJ = J.tolist()

    if interactions is None:
        interactions = {}
        addnew = True

    # now add higher order interactions
    for o in range(2, order + 1):
        for n in np.unique(I):
            muts = J[I == n]

            for combo in combinations(muts, o):
                if addnew and combo not in interactions:
                    interactions[combo] = len(interactions)

                if combo in interactions:
                    II.append(n)
                    JJ.append(interactions[combo] + X.shape[1])

        print(o, len(interactions))

    I = np.array(II)
    J = np.array(JJ)
    D = np.ones(I.shape)

    X_s = sparse.coo_matrix(
        (D, (I, J)), shape=[X.shape[0], len(interactions) + X.shape[1]]
    ).tocsc()

    return X_s, interactions


def partial_lasso(X, y, weights, alpha, sample_weight=None):
    def objective(beta):
        err = np.linalg.norm(X.dot(beta) - y, 2)
        if sample_weight is not None:
            err = err / sample_weight
        return (1.0 / (2 * X.shape[0])) * np.square(err) + alpha * np.linalg.norm(
            weights * beta, 1
        )

    beta0 = np.zeros(X.shape[1])
    beta_hat = minimize(objective, beta0, method="L-BFGS-B", options={"disp": False})
    return beta_hat
