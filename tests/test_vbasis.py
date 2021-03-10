import torch

from lantern.model.basis import VariationalBasis


def test_kl_loss_backward():

    p = 200
    D = 10

    vb = VariationalBasis(p=p, D=D)
    vb(torch.randn(D, p))

    assert vb.W_mu.grad is None
    vb._kl.backward()
    assert vb.W_mu.grad is not None
