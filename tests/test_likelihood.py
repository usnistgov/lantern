import torch
from torch import nn
from torch.distributions import Gamma
from torch.optim import Adam

from lantern.loss import ELBO_GP
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import (
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)


def test_1d_gaussian():

    x = torch.linspace(-1, 1, 100)
    n = torch.rand(100)

    like = GaussianLikelihood()

    nrm = like(x)
    assert (nrm.variance == like.noise).all()

    nrm = like(x, noise=n)
    assert torch.allclose(nrm.variance, like.noise + n)


def test_md_gaussian():

    x = torch.rand(10, 2)
    n = torch.rand(10, 2)

    like = MultitaskGaussianLikelihood(2)

    nrm = like(x)
    assert (nrm.variance == like.task_noises).all()

    nrm = like(x, noise=n)
    assert torch.allclose(nrm.variance, like.task_noises + n)
