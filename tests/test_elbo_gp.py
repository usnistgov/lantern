import torch
from torch import nn
from torch.distributions import Gamma
from torch.optim import Adam

from lantern.loss import ELBO_GP
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import (
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)


def test_factory():

    p = 10
    K = 3
    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )

    D = 1
    m = Model(vb, Phenotype.build(D, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    assert type(elbo.mll.likelihood) == GaussianLikelihood

    D = 2
    m = Model(vb, Phenotype.build(D, K, Ni=100), MultitaskGaussianLikelihood(D))
    elbo = ELBO_GP.fromModel(m, 1000)

    assert type(elbo.mll.likelihood) == MultitaskGaussianLikelihood

    # test parameter building
    Adam(elbo.parameters())


def test_sigma_hoc_grad():

    # one-dim with noise
    p = 10
    K = 3
    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )

    D = 1
    m = Model(vb, Phenotype.build(D, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100,), noise=torch.randn(100).exp())
    total = sum(loss.values())

    assert m.likelihood.raw_noise.grad is None
    total.backward()
    assert m.likelihood.raw_noise.grad is not None

    # one-dim without noise
    m = Model(vb, Phenotype.build(D, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100,))
    total = sum(loss.values())

    assert m.likelihood.raw_noise.grad is None
    total.backward()
    assert m.likelihood.raw_noise.grad is not None

    # multi-dim with noise
    D = 3
    m = Model(vb, Phenotype.build(D, K, Ni=100), MultitaskGaussianLikelihood(3))
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100, D), noise=torch.randn(100, D).exp())
    total = sum(loss.values())

    assert m.likelihood.raw_task_noises.grad is None
    total.backward()
    assert m.likelihood.raw_task_noises.grad is not None

    # multi-dim without noise
    m = Model(vb, Phenotype.build(D, K, Ni=100), MultitaskGaussianLikelihood(3))
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100, D),)
    total = sum(loss.values())

    assert m.likelihood.raw_task_noises.grad is None
    total.backward()
    assert m.likelihood.raw_task_noises.grad is not None
