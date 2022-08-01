from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
import pandas as pd
import numpy as np
import torch

from lantern.model.surface import Functional
from lantern.model import Model
from lantern.model.likelihood import GaussianLikelihood
from lantern.model.basis import VariationalBasis
from lantern.loss import ELBO_GP
from lantern.dataset import Dataset


def test_fxn():

    Nz = 8
    Z = torch.rand(Nz, 1)
    K = 10

    phen = Functional.build(Z, 1, K, Ni=100)

    assert phen.K == K + Z.shape[1]

    assert type(phen.variational_strategy) == VariationalStrategy

    mvn = phen(torch.rand(50, 10))
    assert type(mvn) == MultivariateNormal
    assert mvn.mean.shape == (50 * Nz,)


def test_loss():

    Nz = 8
    Z = torch.rand(Nz, 1)
    K = 10

    phen = Functional.build(Z, 1, K, Ni=100)
    basis = VariationalBasis.build(K=K, p=5)
    like = GaussianLikelihood()

    m = Model(basis, phen, like)

    loss = ELBO_GP.fromModel(m, N=1000)

    X = torch.rand(10, basis.p)
    y = torch.randn(X.shape[0], Nz)

    mvn = m(X)
    lss = loss(mvn, y)

    total = sum(lss.values())

    assert m.likelihood.raw_noise.grad is None
    assert m.basis.W_mu.grad is None
    assert m.surface.kernel.base_kernel.kernels[0].raw_lengthscale.grad is None
    assert m.surface.kernel.base_kernel.kernels[1].raw_lengthscale.grad is None
    total.backward()
    assert m.likelihood.raw_noise.grad is not None
    assert m.basis.W_mu.grad is not None
    assert m.surface.kernel.base_kernel.kernels[0].raw_lengthscale.grad is None
    # this one should have changed:
    assert m.surface.kernel.base_kernel.kernels[1].raw_lengthscale.grad is not None


def test_expand():

    z1 = torch.rand(3, 2)
    z2 = torch.rand(2, 3)

    zexpand = Functional._expand(z1, z2)
    r, c = zexpand.shape
    assert r == z1.shape[0] * z2.shape[0]
    assert c == z1.shape[1] + z2.shape[1]

    for i in range(z1.shape[0]):
        assert torch.allclose(
            zexpand[i * z2.shape[0] : (i + 1) * z2.shape[0], z1.shape[1] :], z2
        )
