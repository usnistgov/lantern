from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy.stats import ks_2samp as ks2
import attr
import matplotlib.pyplot as plt

from lantern.model import Model
from lantern.dataset import Dataset


def _latex_float(f):
    float_str = "{0:.4g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


@attr.s(repr=False)
class Dimensionality:
    logprob: pd.DataFrame = attr.ib(repr=False)
    statistics: pd.DataFrame = attr.ib(repr=False)
    alpha: float = attr.ib(repr=False, default=0.05)

    @property
    def K(self):
        # find the highest dimension where inclusion improves data likelihood
        (thresh,) = np.where(self.statistics.pval < self.alpha)
        K = max(thresh) + 1  # add one to match human expecations

        return K

    def __repr__(self,):
        return f"Dimensionality({self.K})"

    def plotVariance(self, basis, fig_kwargs={"figsize": (4, 3), "dpi": 100}):
        K = basis.K
        qa = 1 / basis.qalpha(detach=True).mean[basis.order]

        plt.figure(**fig_kwargs)
        plt.plot(qa)
        plt.scatter(range(K), qa)

        for k in range(self.K):
            plt.scatter([k], [qa[k]], facecolor="none", color="C0", s=120)

        plt.xticks(range(K))
        plt.semilogy()
        plt.xlabel("dimensions")
        plt.ylabel("variance")

        None

    def plotStatistics(self, nrow=2, fig_kwargs={"dpi": 100}):

        lp = self.logprob
        stat = self.statistics
        K = stat.shape[0]  # total number of dimensions
        ncol = K // nrow

        fig, axes = plt.subplots(
            nrow, K // nrow, figsize=(K // nrow * 2, 2 * nrow), **fig_kwargs
        )
        axes = axes.ravel()

        for k in range(K):
            d0 = lp.filter(regex=f".*k{k}", axis=1).sum(axis=1)
            d1 = lp.filter(regex=f".*k{k+1}", axis=1).sum(axis=1)

            lims = (min(d0.min(), d1.min()), max(d1.max(), d1.max()))
            axes[k].hist(
                d0, label=f"K={k}", alpha=0.6, bins=np.linspace(*lims, 50), log=True,
            )
            axes[k].hist(
                d1, label=f"K={k+1}", alpha=0.6, bins=np.linspace(*lims, 50), log=True,
            )
            axes[k].legend(shadow=True, fancybox=True)
            if k >= K - ncol:
                axes[k].set_xlabel("$E_q[\\log p(y)]$")
            if (k % (K // nrow)) == 0:
                axes[k].set_ylabel("count")
            # axes[k].set_title(f"p = {stat[k].pvalue:.4e}")
            axes[k].set_title(f"$p = {_latex_float(stat.pval[k])}$")

        plt.tight_layout()


def dimensionality(model: Model, dataset: Dataset, alpha=0.05, *args, **kwargs):
    K = model.basis.K
    lp = _logprob_scan(model, dataset, *args, **kwargs)

    stat = pd.DataFrame(
        [
            ks2(
                lp.filter(regex=f".*k{k+1}$", axis=1).sum(axis=1),
                lp.filter(regex=f".*k{k}$", axis=1).sum(axis=1),
            )
            for k in range(K)
        ],
        columns=["score", "pval"],
    )

    return Dimensionality(lp, stat)


def _logprob_scan(
    model: Model, dataset: Dataset, size=1024, resample=1, cuda=False, pbar=False,
):

    D = dataset.D
    K = model.basis.K
    likelihood = model.likelihood

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

            for k in range(0, K + 1):

                samp = torch.zeros(len(_y), D, resample)

                for r in range(resample):
                    _z = model.basis(_x)
                    zpred = torch.zeros_like(_z)

                    if cuda:
                        zpred = zpred.cuda()

                    # copy over dimensions up to the current k
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
