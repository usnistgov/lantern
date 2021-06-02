import string
import pandas as pd
from itertools import combinations
from gpytorch.kernels import RQKernel
import torch
import numpy as np
import matplotlib.pyplot as plt


def sim(seed, p=5):

    N = 2 ** p
    torch.random.manual_seed(seed)
    W = torch.randn(p, 1) * np.sqrt(2)

    X = torch.zeros(N, p)
    ind = 1

    # for all # of mutations
    for mutations in range(1, p + 1):

        # for selected combination of mutations for a variant
        for variant in combinations(range(p), mutations):

            # for each selected
            for s in variant:
                X[ind, s] = 1

            # update after variant
            ind += 1

    z = torch.mm(X, W)
    Z = torch.linspace(z.min(), z.max())[:, None]
    z_samp = torch.cat((z, Z), 0)

    kernel = RQKernel()
    with torch.no_grad():
        K = kernel(z_samp).evaluate()  # + 0.05 * torch.eye(N)
        f = torch.distributions.MultivariateNormal(
            torch.zeros(N + 100), K + torch.eye(N + 100) * 1e-5
        ).rsample()

    y = f[:N] + torch.randn(N) * 0.15

    return W, X, z, y, Z, f[N:]


W, X, z, y, Z, f = sim(100)

plt.plot(Z, f)
plt.scatter(z, y)
