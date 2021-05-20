import pytest
import torch

from gpytorch.kernels import RBFKernel

from lantern.model.surface import Phenotype
from lantern.diffops import robustness, additivity


def test_robustness():

    phen = Phenotype.build(1, 1, Ni=100)
    rob = robustness(phen, torch.randn(100, 1))

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()

    with pytest.raises(ValueError):

        phen = Phenotype.build(1, 10, Ni=100, kernel=RBFKernel())
        rob = robustness(phen, torch.randn(100, 10))


def test_robustness_z0():
    phen = Phenotype.build(1, 10, Ni=100)
    r1 = robustness(phen, torch.randn(100, 10))
    r2 = robustness(
        phen,
        torch.randn(100, 10),
        z0=phen.variational_strategy.inducing_points.detach(),
    )

    assert torch.allclose(r1, r2)


def test_robustness_multidim():
    phen = Phenotype.build(2, 10, Ni=100)
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

    phen = Phenotype.build(1, 10, Ni=100)
    rob = additivity(phen, torch.randn(100, 10))

    assert rob.shape[0] == 100
    assert rob.ndim == 1
    assert (rob >= 0).all()
    assert (rob <= 1).all()

    with pytest.raises(ValueError):

        phen = Phenotype.build(1, 10, Ni=100, kernel=RBFKernel())
        rob = additivity(phen, torch.randn(100, 10))


def test_additivity_multidim():
    phen = Phenotype.build(2, 1, Ni=100)

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
