import attr
import torch
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Gamma, Normal
from torch import nn
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO

from lantern.loss import Term

# hack to allow non-fixed noise for each observation
class _MultitaskGaussianLikelihood(MultitaskGaussianLikelihood):
    def _shaped_noise_covar(self, base_shape, noise=None):
        noise_covar = super()._shaped_noise_covar(base_shape)
        if noise is not None:
            return noise_covar.add_diag(noise)
        return noise_covar


@attr.s
class ELBO_GP(Term):

    """The variational ELBO objective for GPs
    """

    mll = attr.ib(repr=False)
    raw_sigma_hoc = attr.ib(repr=False)

    def loss(self, yhat, y, noise=None, *args, **kwargs) -> dict:

        if noise is not None:
            # fix 1d obseravation, probably needs to be fixed longer tem
            if noise.shape[1] == 1:
                noise = noise[:, 0]
                y = y.reshape(*yhat.mean.shape)

            if self.sigma_hoc:
                noise = noise + F.softplus(self.raw_sigma_hoc)

            # fix for adding to diag
            if self.D > 1:
                noise = noise.reshape(noise.shape[0] * noise.shape[1])

            # note: noise is variance
            ll, kl, log_prior = self.mll(yhat, y, noise=noise)
        else:
            ll, kl, log_prior = self.mll(yhat, y.reshape(*yhat.mean.shape))

        return {
            "neg-loglikelihood": -ll,
            "gp-kl": kl,
            "neg-log-gp-prior": -log_prior,
        }

    @classmethod
    def fromGP(
        cls, gp, N, likelihood=None, objective=VariationalELBO, sigma_hoc_offset=0
    ):
        if likelihood is None:
            likelihood = GaussianLikelihood()
            if gp.D > 1:
                likelihood = _MultitaskGaussianLikelihood(num_tasks=gp.D)

        return cls(
            objective(likelihood, gp, num_data=N, combine_terms=False),
            nn.Parameter(torch.randn(gp.D) + sigma_hoc_offset),
        )
