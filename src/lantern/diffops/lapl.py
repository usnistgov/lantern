from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RQKernel
from gpytorch.lazy import lazify

import torch


def laplacian(model, z, z0=None, dims=None, reduce=True, debug=False, p=0, alpha=None):
    """Calculate the marginal distribution of the laplacian at each point z 

    :param reduce: Whether to reduce the calculation to the laplacian or keep terms expanded for each component (e.g. {d^2f/dx_1^2(z_1)}). Note that the return type will be a multivariate normal, but with a block diagonal covariance matrix for different z_i, representing marginalization b/w different z_i's. Calculations on the distribution will
    :type reduce: bool
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

    if dims is None:
        L = model._get_induc().shape[-1]
        dims = list(range(L))
    else:
        L = len(dims)

    D = 1
    with torch.no_grad():
        qf = model(z0)
        K = model.kernel(z0).add_jitter()
        K = K.evaluate()
        if K.ndim > 2:
            # store number of dims
            D = K.shape[0]
            K = K[p]

        K = lazify(K)

        # build cross-covariance term
        dK = torch.zeros(len(z) * L, len(z0), device=device)

        if alpha is None:
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
        # for i in range(L):
        for ii, i in enumerate(dims):

            # the cross difference in this dimension
            diff = torch.repeat_interleave(
                z[:, [i]], M, dim=1
            ) - torch.repeat_interleave(z0[:, [i]].T, N, dim=0)

            # assign the cross-covariance for this dimension, grouping by input z
            # dK[i::L, :] = -g.pow(-1 - alpha).div(ls[0, i]) - (-1 - alpha) * g.pow(
            #     -2 - alpha
            # ) * diff.pow(2).div(alpha * ls[0, i].pow(2))

            dK[ii::L, :] = -g.pow(-1 - alpha).div(ls[i].pow(2)) - (-1 - alpha) * g.pow(
                -2 - alpha
            ) * diff.pow(2).div(alpha * ls[i].pow(2))

        dK = sigma2 * dK

        # predict partial deriv means
        qfmean = qf.mean
        if qfmean.ndim == 1:
            qfmean = qfmean[:, None]
        else:
            qfmean = qfmean[:, [p]]
        mu = dK.mm(K.inv_matmul(qfmean))

        S = qf.covariance_matrix

        # filter out covariance for multidim, covariance is spaced
        # point-wise, with different dimensions in order
        S = S[:, torch.arange(p, M * D, D)][torch.arange(p, M * D, D), :]

        # COV = K_dd + K_df K_ff^-1 (S - K_ff) K_ff^-1 K_fd
        # K = lazify(
        #     model.variational_strategy.prior_distribution.covariance_matrix[p, :, :]
        # ).add_jitter(1e-4)
        # mt = lazify(S).add_jitter(1e-4) - K
        # mt = (
        #     lazify(S).add_jitter()
        #     - model.variational_strategy.prior_distribution.covariance_matrix[p, :, :]
        # )
        mt = (lazify(S) - K).add_jitter(1e-4)

        mt = K.inv_matmul(mt.evaluate())
        mt = K.inv_matmul(mt.transpose(-1, -2)).transpose(-1, -2)
        # mt = dK @ mt @ dK.transpose(-1, -2)

        # selectively calculate the predictive covariance for each
        # input z position independently
        block = (
            -sigma2
            * (torch.eye(L) * 2 + torch.ones(L, L))
            .to(device)
            .mul(-1 - alpha)
            .div(alpha)
            / torch.repeat_interleave(ls[None, dims].pow(2), L, dim=0)
            / torch.repeat_interleave(ls[None, dims].pow(2).T, L, dim=1)
        )

        cov = torch.zeros(N, L, L, device=device)
        for n in range(N):
            _mt = (
                dK[n * L : (n + 1) * L, :]
                @ mt
                @ dK[n * L : (n + 1) * L, :].transpose(-1, -2)
            )
            cov[n, :, :] = block + _mt

        if debug:
            return mu, cov, dK, mt, block, diff, qf

        # reduce to marginal laplacian values
        if reduce:
            mult = torch.ones(N, 1, L, device=device)
            _mu = torch.bmm(mult, mu.reshape(N, L, 1))
            _var = torch.bmm(torch.bmm(mult, cov,), mult.transpose(-2, -1))

            return _mu[:, 0, 0], _var[:, 0, 0]

        # keep as marginal second derivatives at each point
        return mu.reshape(N, L, 1), cov
