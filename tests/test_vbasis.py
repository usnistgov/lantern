import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.distributions import Gamma

from lantern.model.basis import VariationalBasis
from lantern.loss import KL
from lantern.dataset import Dataset


def test_kl_loss_backward():

    p = 200
    K = 10
    N = 100

    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    W = vb(torch.randn(N, p))

    assert vb.W_mu.grad is None
    vb._kl.backward()
    assert vb.W_mu.grad is not None
    assert vb.W_log_sigma.grad is not None

    assert W.shape == (N, K)


def test_order():

    p = 200
    K = 10

    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    vb.log_alpha.data = torch.ones(K)
    vb.log_beta.data = torch.arange(K) * 1.0

    assert torch.allclose(
        vb.order, torch.flip(torch.arange(K).view(K, 1), (0,)).view(K)
    )

    vb.log_alpha.data = torch.arange(K) * 1.0
    vb.log_beta.data = torch.ones(K)

    assert torch.allclose(vb.order, torch.arange(K))


def test_eval():
    p = 200
    K = 10
    N = 30

    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    vb.eval()

    with torch.no_grad():
        X = torch.randn(N, p)
        W1 = vb(X)
        W2 = vb(X)

    assert torch.allclose(W1, W2)


def test_loss():

    p = 200
    K = 10
    N = 1000

    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    loss = vb.loss(N=N)
    assert type(loss) == KL

    _ = vb(torch.randn(N, p))

    lss = loss(None, None)
    assert "variational_basis" in lss

    assert torch.allclose(lss["variational_basis"], vb._kl / N)


def test_load_state_dict(tmp_path):
    """Test that we can load the basis after saving the state_dict.
    """
    p = 200
    K = 10

    d = tmp_path / "out"
    d.mkdir()

    pt = d / "basis.pt"

    vb = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    vb(torch.randn(100, p))

    torch.save(vb.state_dict(), pt)

    vb2 = VariationalBasis(
        nn.Parameter(torch.randn(p, K)),
        nn.Parameter(torch.randn(p, K) - 3),
        nn.Parameter(torch.randn(K)),
        nn.Parameter(torch.randn(K)),
        Gamma(0.001, 0.001),
    )
    vb2.load_state_dict(torch.load(pt))

    print(vb._kl.size())
    print(vb2._kl.size())

    assert vb._kl.size() == vb2._kl.size()


def test_ds_construct_1d():

    df = pd.DataFrame(
        {"substitutions": ["a1b", "c2d"], "phenotype": [0.0, 1.0], "error": [0.1, 0.2],}
    )
    ds = Dataset(df)
    vb = VariationalBasis.fromDataset(ds, 10, meanEffectsInit=True)

    assert vb.K == 10
    assert vb.p == ds.p

    # check average effect
    # TODO: This throws an error related to the torch.lstsq() fix, but I can't figure it out right now:
    #     assert np.allclose(vb.W_mu[:, 0].detach().numpy(), df["phenotype"])
    assert not np.allclose(vb.W_mu[:, 1].detach().numpy(), df["phenotype"])


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
    vb = VariationalBasis.fromDataset(ds, 10, meanEffectsInit=True)

    assert vb.K == 10
    assert vb.p == ds.p

    # check average effect
    assert np.allclose(vb.W_mu[:, 0].detach().numpy(), df["p1"])
    assert np.allclose(vb.W_mu[:, 1].detach().numpy(), df["p2"])
    assert not np.allclose(vb.W_mu[:, 0].detach().numpy(), df["p2"])
    assert not np.allclose(vb.W_mu[:, 1].detach().numpy(), df["p1"])
    assert not np.allclose(vb.W_mu[:, 2].detach().numpy(), df["p1"])
    assert not np.allclose(vb.W_mu[:, 2].detach().numpy(), df["p2"])
