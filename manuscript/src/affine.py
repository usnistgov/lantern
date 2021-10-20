"""Note: generally not actually allowing for shift in any transformation"""

import torch
import attr
from gpytorch.variational import IndependentMultitaskVariationalStrategy


@attr.s
class Transformation:
    """"""

    K: int = attr.ib()
    i: int = attr.ib()
    j: int = attr.ib()

    def __call__(self, t):
        # _t should be defined in post init method
        return torch.mm(self._t, t)


@attr.s
class Rotation(Transformation):

    theta: float = attr.ib()

    def __attrs_post_init__(self):
        self._t = torch.eye(self.K)
        theta = self.theta
        self._t[self.i, self.i] = torch.cos(torch.deg2rad(theta))
        self._t[self.i, self.j] = -torch.sin(torch.deg2rad(theta))
        self._t[self.j, self.i] = torch.sin(torch.deg2rad(theta))
        self._t[self.j, self.j] = torch.cos(torch.deg2rad(theta))


@attr.s
class Scale(Transformation):

    si: float = attr.ib()
    sj: float = attr.ib()

    def __attrs_post_init__(self):
        self._t = torch.eye(self.K)
        self._t[self.i, self.i] = self.si
        self._t[self.j, self.j] = self.sj


@attr.s
class Shear(Transformation):

    si: float = attr.ib()
    sj: float = attr.ib()

    def __attrs_post_init__(self):
        self._t = torch.eye(self.K)
        self._t[self.i, self.j] = self.si
        self._t[self.j, self.i] = self.sj


def transform(model, *transforms):

    # transform W
    W = model.basis.W_mu.detach()
    for tr in transforms:
        W = tr(W.t()).t()
    model.basis.W_mu.data.copy_(W)

    # update qalpha, from Bishop (1999)
    E = (
        (model.basis.W_log_sigma.exp().pow(2) + model.basis.W_mu.pow(2))
        .sum(axis=0)
        .detach()
    )
    a = model.basis.alpha_prior.concentration + model.basis.W_mu.shape[0] / 2
    b = model.basis.alpha_prior.rate + E / 2
    model.basis.log_alpha.data = torch.log(a.repeat(model.basis.K))
    model.basis.log_beta.data = torch.log(b.reshape(-1))

    # transform inducing points
    strat = model.surface.variational_strategy
    if isinstance(strat, IndependentMultitaskVariationalStrategy):
        strat = strat.base_variational_strategy
    Z = strat.inducing_points.detach()

    for tr in transforms:
        if Z.ndim > 2:
            for i in range(Z.shape[0]):
                Z[i, :, :] = tr(Z[i, :, :].t()).t()
        else:
            Z[:, :] = tr(Z.t()).t()

    Z.data.copy_(Z)  # is this necessary?
