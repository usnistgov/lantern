import torch
from torch import nn
from torch.distributions import Gamma
from torch.optim import Adam
import pandas as pd

from lantern.loss import ELBO_GP
from lantern.model import Model
from lantern.model.basis import VariationalBasis
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

    m = Model(vb, Phenotype.fromDataset(ds_single, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    assert type(elbo.mll.likelihood) == GaussianLikelihood

    m = Model(vb, Phenotype.fromDataset(ds_multi, K, Ni=100), MultitaskGaussianLikelihood(ds_multi.D))
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

    m = Model(vb, Phenotype.fromDataset(ds_single, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100,), noise=torch.randn(100).exp())
    total = sum(loss.values())

    assert m.likelihood.raw_noise.grad is None
    total.backward()
    assert m.likelihood.raw_noise.grad is not None

    # one-dim without noise
    m = Model(vb, Phenotype.fromDataset(ds_single, K, Ni=100), GaussianLikelihood())
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100,))
    total = sum(loss.values())

    assert m.likelihood.raw_noise.grad is None
    total.backward()
    assert m.likelihood.raw_noise.grad is not None

    # multi-dim with noise
    m = Model(vb, Phenotype.fromDataset(ds_multi, K, Ni=100), MultitaskGaussianLikelihood(ds_multi.D))
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100, ds_multi.D), noise=torch.randn(100, ds_multi.D).exp())
    total = sum(loss.values())

    assert m.likelihood.raw_task_noises.grad is None
    total.backward()
    assert m.likelihood.raw_task_noises.grad is not None

    # multi-dim without noise
    m = Model(vb, Phenotype.fromDataset(ds_multi, K, Ni=100), MultitaskGaussianLikelihood(ds_multi.D))
    elbo = ELBO_GP.fromModel(m, 1000)

    yhat = m.surface(torch.randn(100, K))
    loss = elbo(yhat, torch.randn(100, ds_multi.D),)
    total = sum(loss.values())

    assert m.likelihood.raw_task_noises.grad is None
    total.backward()
    assert m.likelihood.raw_task_noises.grad is not None
