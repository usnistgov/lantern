import pytest
import torch

from lantern.model import Model
from lantern.model.basis import Basis, VariationalBasis
from lantern.model.surface import Phenotype


def test_model_validator():

    with pytest.raises(ValueError):
        Model(Basis(p=10, K=3), Phenotype(4, torch.randn(4, 100, 5)))


def test_forward():

    m = Model(VariationalBasis(p=10, K=3), Phenotype(4, torch.randn(4, 100, 3)))
    m.eval()

    X = torch.randn(30, 10)
    out = m(X)
    out2 = m.surface(m.basis(X))

    assert torch.allclose(out.mean, out2.mean)


def test_loss():
    m = Model(VariationalBasis(p=10, K=3), Phenotype(4, torch.randn(4, 100, 3)))
    loss = m.loss(N=1000)

    X = torch.randn(30, 10)
    yhat = m(X)
    lss = loss(yhat, torch.randn(30, 4))

    assert "variational_basis" in lss
    assert "neg-loglikelihood" in lss
