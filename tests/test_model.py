import pytest
import torch
from torch import nn
from torch.distributions import Gamma
import pandas as pd

from lantern.model import Model
from lantern.model.basis import Basis, VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import (
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)


from lantern.dataset import Dataset

df = pd.DataFrame({'substitutions':['A', 'B', 'A:B'], 'phen_0':[1,2,3], 'phen_0_var':[1,1,1]})
ds_single = Dataset(df, phenotypes = ['phen_0'], errors = ['phen_0_var'])

df_m = df.copy()
df_m['phen_1'] = [2,4,6]
df_m['phen_1_var'] = [2,2,2]
ds_multi = Dataset(df_m, phenotypes = ['phen_0', 'phen_1'], errors = ['phen_0_var', 'phen_1_var'])


def test_model_validator():
    class DummyBasis(Basis):
        @property
        def p(self):
            return 10

        @property
        def K(self):
            return 3

    with pytest.raises(ValueError):
        Model(DummyBasis(), Phenotype.fromDataset(ds_multi, 5,), MultitaskGaussianLikelihood(5))


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

    m = Model(vb, Phenotype.fromDataset(ds_multi, K, Ni=100), GaussianLikelihood())
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

    m = Model(vb, Phenotype.fromDataset(ds_multi, 3, Ni=100), MultitaskGaussianLikelihood(ds_multi.D))
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
