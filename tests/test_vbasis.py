import torch

from lantern.model.basis import VariationalBasis


def test_kl_loss_backward():

    p = 200
    K = 10
    N = 100

    vb = VariationalBasis(p=p, K=K)
    W = vb(torch.randn(N, p))

    assert vb.W_mu.grad is None
    vb._kl.backward()
    assert vb.W_mu.grad is not None
    assert vb.W_log_sigma.grad is not None

    assert W.shape == (N, K)


def test_order():

    p = 200
    K = 10

    vb = VariationalBasis(p=p, K=K)
    vb.log_alpha.data = torch.ones(K)
    vb.log_beta.data = torch.arange(K) * 1.0

    assert torch.allclose(
        vb.order, torch.flip(torch.arange(K).view(K, 1), (0,)).view(K)
    )


def test_eval():
    p = 200
    K = 10
    N = 30

    vb = VariationalBasis(p=p, K=K)
    vb.eval()

    with torch.no_grad():
        X = torch.randn(N, p)
        W1 = vb(X)
        W2 = vb(X)

    assert torch.allclose(W1, W2)
