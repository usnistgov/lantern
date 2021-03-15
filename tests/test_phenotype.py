from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean, Mean
from gpytorch.kernels import Kernel, ScaleKernel, RQKernel
import pandas as pd
import numpy as np
import torch
import pytest

from lantern.model.surface import Phenotype
from lantern.loss import ELBO_GP
from lantern.dataset import Dataset


def test_1d():

    induc = torch.rand(100, 10)
    phen = Phenotype(1, induc)

    assert type(phen.variational_strategy) == VariationalStrategy

    mvn = phen(torch.rand(50, 10))
    assert type(mvn) == MultivariateNormal
    assert mvn.mean.shape == (50,)

    assert np.allclose(induc.numpy(), phen._get_induc())

    induc = torch.rand(100, 10)
    phen._set_induc(induc.numpy())
    assert np.allclose(induc.numpy(), phen._get_induc())


def test_multid():

    induc = torch.rand(4, 100, 10)
    phen = Phenotype(4, induc)

    assert type(phen.variational_strategy) == IndependentMultitaskVariationalStrategy

    mvn = phen(torch.rand(50, 10))
    assert type(mvn) == MultitaskMultivariateNormal
    assert mvn.mean.shape == (50, 4)

    assert np.allclose(induc.numpy(), phen._get_induc())

    induc = torch.rand(100, 10)
    phen._set_induc(induc.numpy())
    assert np.allclose(induc.numpy(), phen._get_induc())

    assert not phen.kernel.base_kernel.lengthscale.requires_grad

    with pytest.raises(ValueError):
        Phenotype(4, torch.randn(100, 10))


def test_loss():

    induc = torch.rand(100, 10)
    phen = Phenotype(1, induc)
    loss = phen.loss(N=1000)
    assert type(loss) == ELBO_GP

    mvn = phen(torch.randn(50, 10))

    lss = loss(mvn, torch.randn(50))
    assert "neg-loglikelihood" in lss
    assert "neg-log-gp-prior" in lss
    assert "gp-kl" in lss


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
