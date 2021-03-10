import torch

from lantern.model.loss import Loss, Term


def test_loss_agg():
    class A(Term):
        def loss(*args, **kwargs):
            return {"a": torch.randn(1)}

    class B(Term):
        def loss(*args, **kwargs):
            return {"b": torch.randn(1)}

    loss = Loss([A(), B()])
    lss = loss.loss(None, None)
    assert "a" in lss
    assert "b" in lss
