import pytest
import torch
import pandas as pd

from gpytorch.kernels import RBFKernel

from lantern.model.surface import Phenotype
from lantern.diffops import robustness, additivity
from lantern.dataset import Dataset


df = pd.DataFrame({'substitutions':['A', 'B', 'A:B'], 'phen_0':[1,2,3], 'phen_0_var':[1,1,1]})
ds_single = Dataset(df, phenotypes = ['phen_0'], errors = ['phen_0_var'])

df_m = df.cop()
df_m['phen_1'] = [2,4,6]
df_m['phen_1_var'] = [2,2,2]
ds_multi = Dataset(df_m, phenotypes = ['phen_0', 'phen_1'], errors = ['phen_0_var', 'phen_1_var'])


def test_robustness():
    phen = Phenotype.build(ds_single.D, ds_single, 1, Ni=100)
    rob = robustness(phen, torch.randn(100, 1))

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()

    with pytest.raises(ValueError):
        
        phen = Phenotype.build(ds_multi.D, ds_multi, 10, Ni=100, kernel=RBFKernel())
        rob = robustness(phen, torch.randn(100, 10))


def test_robustness_z0():
    phen = Phenotype.build(ds_single.D, ds_single, 10, Ni=100)
    r1 = robustness(phen, torch.randn(100, 10))
    r2 = robustness(
        phen,
        torch.randn(100, 10),
        z0=phen.variational_strategy.inducing_points.detach(),
    )

    assert torch.allclose(r1, r2)


def test_robustness_multidim():
    phen = Phenotype.build(ds_multi.D, ds_multi, 10, Ni=100)
    rob = robustness(phen, torch.randn(100, 10))

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()

    rob = robustness(phen, torch.randn(100, 10), p=1)

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()


def test_additivity():
    phen = Phenotype.build(ds_single.D, ds_single, 10, Ni=100)
    rob = additivity(phen, torch.randn(100, 10))

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()

    with pytest.raises(ValueError):

        phen = Phenotype.build(ds_single.D, ds_single, 10, Ni=100, kernel=RBFKernel())
        rob = additivity(phen, torch.randn(100, 10))


def test_additivity_multidim():
    phen = Phenotype.build(ds_multi.D, ds_multi, 1, Ni=100)

    a1 = additivity(phen, torch.randn(100, 1))

    assert a1.shape[0] == 100
    assert a1.ndim == 1
    assert (a1 >= 0).all()
    assert (a1 <= 1).all()

    a2 = additivity(phen, torch.randn(100, 1), p=1)

    assert a2.shape[0] == 100
    assert a2.ndim == 1
    assert (a2 >= 0).all()
    assert (a2 <= 1).all()
