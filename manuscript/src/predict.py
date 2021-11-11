from torch.utils.data import DataLoader
import numpy as np
import gpytorch
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
import pandas as pd

# from src.analyze.grad import gradient, laplacian
# from src.model import LatentLinearGPBayes


def predictions(
    D, model, dataset, size=32, cuda=False, pbar=False, uncertainty=False, dimScan=False
):

    embed = hasattr(model, "L")
    diffops = False  # isinstance(model, LatentLinearGPBayes) and model.D == 1

    # prep for predictions
    loader = DataLoader(dataset, size)
    yhat = torch.zeros(len(dataset), D)
    y = torch.zeros(len(dataset), D)
    noise = torch.zeros(len(dataset), D)
    lp = torch.zeros(len(dataset), D)
    yhat_std = torch.zeros(len(dataset), D)
    if dimScan:
        yhat_scan = torch.zeros(len(dataset), model.L * D)

    if diffops:
        grad_mu = torch.zeros(len(dataset), model.L)
        grad_var = torch.zeros(len(dataset), model.L)
        lapl_mu = torch.zeros(len(dataset), 1)
        lapl_var = torch.zeros(len(dataset), 1)
        z0 = torch.from_numpy(model.landscape._get_induc())

        # magic number, consider changing
        dims = sum(
            (model.embed.log_beta.exp() / (model.embed.log_alpha.exp() - 1)) > 1e-2
        )
        dims = model.embed.variance_order[:dims]

    if embed:
        z = torch.zeros(len(dataset), model.L)

    if cuda:
        yhat = yhat.cuda()
        y = y.cuda()
        lp = lp.cuda()
        if embed:
            z = z.cuda()
        yhat_std = yhat_std.cuda()

        if diffops:
            grad_mu = grad_mu.cuda()
            grad_var = grad_var.cuda()
            lapl_mu = lapl_mu.cuda()
            lapl_var = lapl_var.cuda()

            z0 = z0.cuda()

    # loop over data and generate predictions
    i = 0
    loop = tqdm(loader) if pbar else loader
    for btch in loop:
        _x, _y = btch[:2]
        _x = _x.float()
        if cuda:
            _x = _x.cuda()
            _y = _y.cuda()

        with torch.no_grad():
            _yh = model(_x)

            if embed:
                _z = model.embed(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]

            if diffops:
                _grad = gradient(model.landscape, _z, z0)
                _lapl = laplacian(model.landscape, _z, z0, dims=dims)

            if uncertainty:
                Nsamp = 50
                tmp = torch.zeros(Nsamp, *_y.shape)
                model.train()
                for n in range(Nsamp):
                    f = model(_x)
                    samp = f.sample()
                    if samp.ndim == 1:
                        samp = samp[:, None]

                    tmp[n, :, :] = samp

                yhat_std[i : len(_y) + i, :] = tmp.std(axis=0)

                model.eval()

            # check prediction accuracy as a function of available
            # latent dimensions
            if dimScan:
                _z = model.basis(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]

                for l in range(model.L):
                    _zz = torch.zeros_like(_z)

                    # copy lth first dimensions
                    _zz[:, model.embed.variance_order[:l]] = _z[
                        :, model.embed.variance_order[:l]
                    ]
                    _yyh = model.landscape(_zz)

                    # get to actual prediction
                    if isinstance(_yyh, tuple):
                        _yyh = _yyh[0]

                    if isinstance(_yyh, gpytorch.distributions.MultivariateNormal):
                        _yyh = _yyh.mean

                    if _yyh.ndim == 1:
                        _yyh = _yyh[:, None]

                    yhat_scan[i : len(_y) + i, l * D : (l + 1) * D] = _yyh

        # filter out extra output
        if isinstance(_yh, tuple):
            _yh = _yh[0]

        # need to get a mean prediction, and we can get logprob
        if isinstance(_yh, gpytorch.distributions.MultivariateNormal):
            # # get predictive observation likelihood
            # _yh = model.mll.likelihood(_yh)

            # convert to 1d for individual observations
            # norm = torch.distributions.Normal(
            #     _yh.mean.view(-1, D), torch.sqrt(_yh.variance.view(-1, D))
            # )

            # # update values
            # _lp = norm.log_prob(_y).detach()
            # _lp = _lp.view(-1, D)
            # lp[i : len(_y) + i, :] = _lp
            _yh = _yh.mean

        _y = _y.view(-1, D)
        _yh = _yh.view(-1, D)
        y[i : len(_y) + i, :] = _y
        yhat[i : len(_y) + i, :] = _yh

        if embed:
            z[i : len(_y) + i, :] = _z

        if diffops:
            grad_mu[i : len(_y) + i, :] = _grad[0][:, :, 0]
            grad_var[i : len(_y) + i, :] = _grad[1][
                :, np.arange(model.L), np.arange(model.L)
            ]
            lapl_mu[i : len(_y) + i, 0] = _lapl[0]
            lapl_var[i : len(_y) + i, 0] = _lapl[1]

        # grab noise if available
        if len(btch) > 2:
            _n = btch[2]
            if cuda:
                _n = _n.cuda()
            noise[i : len(_n) + i, :] = _n

        i += len(_x)

    # prep for returning
    y = y.cpu().numpy()
    yhat = yhat.cpu().numpy()
    lp = lp.cpu().numpy()
    noise = noise.cpu().numpy()

    if embed:
        if hasattr(model.embed, "variance_order"):
            z = z[:, model.embed.variance_order]
        z = z.cpu().numpy()

    if diffops:
        grad_mu = grad_mu.cpu().numpy()
        grad_var = grad_var.cpu().numpy()
        lapl_mu = lapl_mu.cpu().numpy()
        lapl_var = lapl_var.cpu().numpy()

    # ret = dict(y=y, yhat=yhat, logprob=lp, noise=noise,)
    ret = dict(y=y, yhat=yhat, noise=noise,)

    if embed:
        ret["z"] = z

    if diffops:
        ret["grad_mu"] = grad_mu
        ret["grad_var"] = grad_var
        ret["lapl_mu"] = lapl_mu
        ret["lapl_var"] = lapl_var

    if uncertainty:
        ret["yhat_std"] = yhat_std.cpu().numpy()

    if dimScan:
        for l in range(model.L):
            ret[f"yhat_d{l}_"] = yhat_scan[:, l * D : (l + 1) * D].cpu().numpy()

    return ret


def cv_scores(pths, func_score=False, noiseless=False, i=0, metric=r2_score):
    """ Calculate the cv scores for a set of predictions
    """

    __scores = pd.concat([pd.read_csv(pths.format(c=i)) for i in range(10)])

    if func_score:
        __scores = (
            __scores.groupby("cv")
            .apply(
                lambda x: metric(
                    x.observed_phenotype,
                    x.func_score,
                    sample_weight=None if noiseless else 1 / x.func_score_var,
                )
            )
            .to_frame("metric")
        )
    else:
        __scores = (
            __scores.groupby("cv")
            .apply(
                lambda x: metric(
                    x[f"y{i}"],
                    x[f"yhat{i}"],
                    sample_weight=None if noiseless else 1 / x[f"noise{i}"],
                )
            )
            .to_frame("metric")
        )

    return __scores


def logprob_scan(
    D, K, model, likelihood, dataset, size=32, resample=1, cuda=False, pbar=False,
):

    # prep for predictions
    loader = DataLoader(dataset, size)
    lp = torch.zeros(len(dataset), D * (K + 1))
    lps = torch.zeros(len(dataset), D * (K + 1))

    if cuda:
        lp = lp.cuda()
        lps = lps.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    # loop over data and generate predictions
    i = 0
    loop = tqdm(loader) if pbar else loader
    for btch in loop:
        _x, _y = btch[:2]
        if _y.ndim == 1:
            _y = _y[:, [0]]

        _x = _x.float()
        _n = btch[2] if len(btch) > 2 else torch.zeros_like(_y)
        if cuda:
            _x = _x.cuda()
            _y = _y.cuda()
            _n = _n.cuda()

        with torch.no_grad():
            # _z = model.basis(_x)

            for k in range(0, K + 1):

                samp = torch.zeros(len(_y), D, resample)

                for r in range(resample):
                    _z = model.basis(_x)
                    zpred = torch.zeros_like(_z)

                    if cuda:
                        zpred = zpred.cuda()
                    for kk in model.basis.order[:k]:
                        zpred[:, kk] = _z[:, kk]

                    _yh = model.surface(zpred)

                    tmp = _yh.__class__(
                        _yh.mean,
                        likelihood._shaped_noise_covar(
                            _yh.mean.shape, noise=_n.reshape(-1)
                        )
                        + _yh.covariance_matrix,
                    )

                    norm = torch.distributions.Normal(
                        _yh.mean.view(-1, D), torch.sqrt(tmp.variance.view(-1, D))
                    )

                    samp[:, :, r] = norm.log_prob(_y).detach().view(-1, D)
                    # samp[:, :, r] = (_y - _yh.mean) ** 2

                # _lp = norm.log_prob(_y).detach()
                # _lp = _lp.view(-1, D)
                _lp = torch.mean(samp, dim=2)

                lp[i : len(_y) + i, k * D : (k + 1) * D] = _lp
                lps[i : len(_y) + i, k * D : (k + 1) * D] = torch.std(samp, dim=2)

        i += len(_x)

    # prep for returning
    lp = lp.cpu().numpy()
    lps = lps.cpu().numpy()

    ret = {}
    for k in range(K + 1):
        for d in range(D):
            ret[f"lp{d}-k{k}"] = lp[:, k * D + d]
            ret[f"lp{d}-k{k}-std"] = lps[:, k * D + d]

    return pd.DataFrame(ret)
