import pytest
import torch
from torch import nn
from torch.distributions import Gamma

from lantern.model import Model
from lantern.model.basis import Basis, VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import (
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)


def test_model_validator():
    class DummyBasis(Basis):
        @property
        def p(self):
            return 10

        @property
        def K(self):
            return 3

    with pytest.raises(ValueError):
        Model(DummyBasis(), Phenotype.build(4, 5,), MultitaskGaussianLikelihood(5))


def test_forward():

    p = 10
    K = 3
    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )

    m = Model(vb, Phenotype.build(4, K, Ni=100), GaussianLikelihood())
    m.eval()

    X = torch.randn(30, 10)
    out = m(X)
    out2 = m.surface(m.basis(X))

    assert torch.allclose(out.mean, out2.mean)


def test_loss():

    p = 10
    K = 3
    D = 4
    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )

    m = Model(vb, Phenotype.build(D, 3, Ni=100), MultitaskGaussianLikelihood(D))
    loss = m.loss(N=1000)

    X = torch.randn(30, 10)
    yhat = m(X)
    lss = loss(yhat, torch.randn(30, 4))

    assert "variational_basis" in lss
    assert "neg-loglikelihood" in lss
    assert "neg-log-gp-prior" in lss
    assert "gp-kl" in lss


# def test_loss():
#
#     phen = Phenotype.build(1, 10, Ni=100)
#     loss = phen.loss(N=1000)
#     assert type(loss) == ELBO_GP
#
#     mvn = phen(torch.randn(50, 10))
#
#     lss = loss(mvn, torch.randn(50))
#     assert "neg-loglikelihood" in lss
#     assert "neg-log-gp-prior" in lss
#     assert "gp-kl" in lss
