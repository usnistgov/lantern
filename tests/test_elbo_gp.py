import torch
from gpytorch.likelihoods import GaussianLikelihood

from lantern.model.surface import Phenotype
from lantern.loss import ELBO_GP
from lantern.loss.elbo_gp import _MultitaskGaussianLikelihood


def test_factory():

    surf = Phenotype.build(1, K=5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000)

    assert type(elbo.mll.likelihood) == GaussianLikelihood

    surf = Phenotype.build(2, K=5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000)

    assert type(elbo.mll.likelihood) == _MultitaskGaussianLikelihood

    from torch.optim import Adam

    Adam(elbo.parameters())


def test_sigma_hoc_grad():

    # one-dim with noise
    surf = Phenotype.build(1, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=True)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100,), noise=torch.randn(100, 1).exp())
    total = sum(loss.values())

    assert elbo.raw_sigma_hoc.grad is None
    total.backward()
    assert elbo.raw_sigma_hoc.grad is not None

    # one-dim without noise
    surf = Phenotype.build(1, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=True)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100,))
    total = sum(loss.values())

    assert elbo.raw_sigma_hoc.grad is None
    total.backward()
    assert elbo.raw_sigma_hoc.grad is None

    # multi-dim with noise
    surf = Phenotype.build(3, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=True)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100, 3), noise=torch.randn(100, 3).exp())
    total = sum(loss.values())

    assert elbo.raw_sigma_hoc.grad is None
    total.backward()
    assert elbo.raw_sigma_hoc.grad is not None

    # multi-dim without noise
    surf = Phenotype.build(3, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=True)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100, 3),)
    total = sum(loss.values())

    assert elbo.raw_sigma_hoc.grad is None
    total.backward()
    assert elbo.raw_sigma_hoc.grad is None


def test_no_sigma_hoc_grad():

    # one-dim
    surf = Phenotype.build(1, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=False)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100,), noise=torch.randn(100, 1).exp())
    total = sum(loss.values())

    assert surf.variational_strategy.inducing_points.grad is None
    total.backward()
    assert surf.variational_strategy.inducing_points.grad is not None

    # multi-dim
    surf = Phenotype.build(3, 5, Ni=100)
    elbo = ELBO_GP.fromGP(surf, 1000, sigma_hoc=True)

    yhat = surf(torch.randn(100, 5))
    loss = elbo(yhat, torch.randn(100, 3), noise=torch.randn(100, 3).exp())
    total = sum(loss.values())

    assert (
        surf.variational_strategy.base_variational_strategy.inducing_points.grad is None
    )
    total.backward()
    assert (
        surf.variational_strategy.base_variational_strategy.inducing_points.grad
        is not None
    )
