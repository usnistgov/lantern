import torch

from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RQKernel
from gpytorch.lazy import lazify


def gradient(model, z, z0=None, batchsize=1024, p=0):
    """Calculate the marginal distribution of the gradient at each point z.
    
    :param model: A `~lantern.model.Surface` to calculate the gradient for
    :type model: lantern.model.Surface
    :param torch.tensor z: positions to calculate the gradient for
    :param torch.tensor z0: reference positions for calculating the conditional distribution of the gradient, defaults to the surface inducing points
    """

    N, L = z.shape

    device = z.device
    mu = torch.zeros(N, L, 1, device=device)
    cov = torch.zeros(N, L, L, device=device)

    for n in range(0, N, batchsize):
        _mu, _cov = _gradient(model, z[n : n + 1024, :], z0, p=p)
        mu[n : n + 1024, :, :] = _mu
        cov[n : n + 1024, :, :] = _cov

    return mu, cov


def _gradient(model, z, z0=None, p=0):
    """Calculate the marginal distribution of the gradient at each point z 
    """

    if not isinstance(model.kernel, ScaleKernel) or not isinstance(
        model.kernel.base_kernel, RQKernel
    ):
        raise ValueError(
            "Cannot compute gradients from this kernel ({})!".format(model.kernel)
        )

    if z0 is None:
        z0 = z

    device = z.device

    L = z.shape[1]
    dims = list(range(L))

    D = 1
    with torch.no_grad():
        qf = model(z0)
        qfmean = qf.mean
        if qfmean.ndim == 1:
            qfmean = qfmean[:, None]
        else:
            qfmean = qfmean[:, [p]]

        K = model.kernel(z0).add_jitter()
        K = K.evaluate()
        if K.ndim > 2:
            # store number of dims
            D = K.shape[0]
            K = K[p]
        K = lazify(K)

        alpha = model.kernel.base_kernel.alpha[p]
        sigma2 = model.kernel.outputscale
        if sigma2.ndim > 0:
            sigma2 = sigma2[p]
        ls = model.kernel.base_kernel.lengthscale[p].reshape(-1)

        z = z.div(ls)
        z0 = z0.div(ls)

        dist = model.kernel.covar_dist(z, z0, square_dist=True)
        g = 1 + dist.div(2 * alpha)

        N = z.shape[0]
        M = z0.shape[0]

        # calculate cross covariance for each point and each dim
        dK = torch.zeros(N, L, len(z0), device=device)
        for ii, i in enumerate(dims):

            # the cross difference in this dimension
            diff = torch.repeat_interleave(
                z[:, [i]], M, dim=1
            ) - torch.repeat_interleave(z0[:, [i]].T, N, dim=0)

            # assign the cross-covariance for this dimension, grouping by input z
            dK[:, ii, :] = -g.pow(-1 - alpha).div(ls[i]).mul(diff)

        dK = sigma2 * dK

        # filter out covariance for multidim, covariance is spaced point-wise, with different dimensions
        S = qf.covariance_matrix
        S = S[:, torch.arange(p, M * D, D)][torch.arange(p, M * D, D), :]

        # COV = K_dd + K_df K_ff^-1 (S - K_ff) K_ff^-1 K_fd
        # mt = lazify(qf.covariance_matrix).add_jitter() - K
        mt = lazify(S).add_jitter() - K
        mt = K.inv_matmul(mt.evaluate())
        mt = K.inv_matmul(mt.transpose(-1, -2)).transpose(-1, -2)

        # selectively calculate the predictive covariance for each
        # input z position independently
        block = sigma2 * torch.diag(1 / ls.pow(2))

        # mu = torch.matmul(dK, K.inv_matmul(qf.mean[:, None])[None, :, :])
        mu = torch.matmul(dK, K.inv_matmul(qfmean)[None, :, :])
        cov = block + torch.bmm(dK, torch.matmul(mt, dK.transpose(-1, -2)),)

        # keep as marginal second derivatives at each point
        return mu, cov
