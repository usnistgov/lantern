from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RQKernel
from gpytorch.lazy import lazify

import torch

from lantern.diffops.grad import gradient
from lantern.diffops.lapl import laplacian


def mdist(mu, cov, z=None):
    """Mahalanobis distance.
    """

    if mu.ndim == 1:
        mu = mu.reshape(-1, 1, 1)
    elif mu.ndim == 2:
        mu = mu.reshape(*mu.shape, 1)

    if cov.ndim == 1:
        cov = cov.reshape(-1, 1, 1)
    elif cov.ndim == 2:
        cov = cov.reshape(*cov.shape, 1)

    if z is None:
        z = torch.zeros_like(mu)

    diff = mu - z

    return (
        torch.bmm(torch.bmm(diff.transpose(-2, -1), torch.inverse(cov)), diff).reshape(
            -1
        )
        ** 0.5
    )


def kernel(mu, cov, z=None):
    """The unnormalized density, or kernel, of the provided value z (default 0).
    """

    dsq = mdist(mu, cov, z) ** 2
    return torch.exp(-0.5 * dsq)


def robustness(surface, z, *args, **kwargs):
    """Calculate the surface robustness at each position z.
    """

    mu, cov = gradient(surface, z, *args, **kwargs)
    return kernel(mu, cov)


def additivity(surface, z, *args, **kwargs):
    """Calculate the surface additivity at each position z.
    """

    mu, cov = laplacian(surface, z, *args, **kwargs)
    return kernel(mu, cov)
