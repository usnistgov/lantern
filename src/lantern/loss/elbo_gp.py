import attr
import torch
from gpytorch.mlls import VariationalELBO

from lantern.loss import Term


@attr.s(eq=False)
class ELBO_GP(Term):

    """The variational ELBO objective for GPs
    """

    mll = attr.ib(repr=False)

    def loss(self, yhat, y, noise=None, *args, **kwargs) -> dict:
        shape = yhat.mean.shape
        y = y.reshape(*shape)

        if noise is not None:
            noise = noise.reshape(*shape)

        if y.isnan().any():
            # in order to support D>1, need to make selection with mask on yhat work
            if yhat.mean.ndim > 1:
                raise ValueError("No support for masking with D>1")

            mask = ~y.isnan()
            imask = torch.where(mask)

            y = y[mask]
            yhat = yhat.__getitem__(imask)

            if noise is not None:
                noise = noise[mask]

        ll, kl, log_prior = self.mll(yhat, y, noise=noise, **kwargs)

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
