from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
import pandas as pd
import numpy as np
import torch

from lantern.model.surface import Phenotype
from lantern.loss import ELBO_GP
from lantern.dataset import Dataset

df = pd.DataFrame({'substitutions':['A', 'B', 'A:B'], 'phen_0':[1,2,3], 'phen_0_var':[1,1,1]})
ds_single = Dataset(df, phenotypes = ['phen_0'], errors = ['phen_0_var'])

df_m = df.copy()
df_m['phen_1'] = [2,4,6]
df_m['phen_1_var'] = [2,2,2]
ds_multi = Dataset(df_m, phenotypes = ['phen_0', 'phen_1'], errors = ['phen_0_var', 'phen_1_var'])


def test_1d():

    phen = Phenotype.fromDataset(ds_single, 10, Ni=100)

    assert type(phen.variational_strategy) == VariationalStrategy

    mvn = phen(torch.rand(50, 10))
    assert type(mvn) == MultivariateNormal
    assert mvn.mean.shape == (50,)

    induc = torch.rand(100, 10)
    assert not np.allclose(induc.numpy(), phen._get_induc())
    phen._set_induc(induc.numpy())
    assert np.allclose(induc.numpy(), phen._get_induc())


def test_multid():

    phen = Phenotype.fromDataset(ds_multi, 10, Ni=100)

    assert type(phen.variational_strategy) == IndependentMultitaskVariationalStrategy

    mvn = phen(torch.rand(50, 10))
    assert type(mvn) == MultitaskMultivariateNormal
    assert mvn.mean.shape == (50, 4)

    induc = torch.rand(4, 100, 10)
    assert not np.allclose(induc.numpy(), phen._get_induc())
    phen._set_induc(induc.numpy())
    assert np.allclose(induc.numpy(), phen._get_induc())

    assert not phen.kernel.base_kernel.lengthscale.requires_grad


def test_ds_construct_1d():

    df = pd.DataFrame(
        {"substitutions": ["a1b", "c2d"], "phenotype": [0.0, 1.0], "error": [0.1, 0.2],}
    )
    ds = Dataset(df)
    phen = Phenotype.fromDataset(ds, 10)

    assert phen.K == 10
    assert phen.D == 1


def test_ds_construct_multid():

    df = pd.DataFrame(
        {
            "substitutions": ["a1b", "c2d"],
            "p1": [0.0, 1.0],
            "p2": [1.0, 0.0],
            "e1": [0.1, 0.2],
            "e2": [0.2, 0.1],
        }
    )

    ds = Dataset(df, phenotypes=["p1", "p2"], errors=["e1", "e2"])
    phen = Phenotype.fromDataset(ds, 10)

    assert phen.K == 10
    assert phen.D == 2
