import pytest
import torch
from torch import nn
from torch.distributions import Gamma

from lantern.model import Model
from lantern.model.basis import Basis, VariationalBasis
from lantern.model.surface import Phenotype


def test_model_validator():
    class DummyBasis(Basis):
        @property
        def p(self):
            return 10

        @property
        def K(self):
            return 3

    with pytest.raises(ValueError):
        Model(DummyBasis(), Phenotype(4, torch.randn(4, 100, 5)))


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

    m = Model(vb, Phenotype(4, torch.randn(4, 100, 3)))
    m.eval()

    X = torch.randn(30, 10)
    out = m(X)
    out2 = m.surface(m.basis(X))

    assert torch.allclose(out.mean, out2.mean)


def test_loss():

    p = 10
    K = 3
    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )

    m = Model(vb, Phenotype(4, torch.randn(4, 100, 3)))
    loss = m.loss(N=1000)

    X = torch.randn(30, 10)
    yhat = m(X)
    lss = loss(yhat, torch.randn(30, 4))

    assert "variational_basis" in lss
    assert "neg-loglikelihood" in lss
