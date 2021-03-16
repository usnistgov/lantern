import torch

from lantern.loss import Composite, Term


def test_loss_agg():
    class A(Term):
        def loss(*args, **kwargs):
            return {"a": torch.randn(1)}

    class B(Term):
        def loss(*args, **kwargs):
            return {"b": torch.randn(1)}

    loss = Composite([A(), B()])
    lss = loss.loss(None, None)
    assert "a" in lss
    assert "b" in lss

    loss = A() + B()
    assert type(loss) == Composite
    lss = loss.loss(None, None)
    assert "a" in lss
    assert "b" in lss

    loss = Composite([A()]) + B()
    assert type(loss) == Composite
    lss = loss.loss(None, None)
    assert "a" in lss
    assert "b" in lss

    loss = B() + Composite([A()])
    assert type(loss) == Composite
    lss = loss.loss(None, None)
    assert "a" in lss
    assert "b" in lss
