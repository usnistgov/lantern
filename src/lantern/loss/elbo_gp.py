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


@attr.s(eq=False)
class ELBO_GP(Term):

    """The variational ELBO objective for GPs
    """

    mll = attr.ib(repr=False)

    def loss(self, yhat, y, noise=None, *args, **kwargs) -> dict:
        ll, kl, log_prior = self.mll(yhat, y, noise=noise)

        return {
            "neg-loglikelihood": -ll,
            "gp-kl": kl,
            "neg-log-gp-prior": -log_prior,
        }

    @classmethod
    def fromModel(
        cls, model, N, objective=VariationalELBO,
    ):
        return cls(
            objective(model.likelihood, model.surface, num_data=N, combine_terms=False),
        )
