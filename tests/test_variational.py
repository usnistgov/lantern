import torch

from lantern.model import Variational


def test_variational_stores_kl():
    class Test(Variational):
        def _forward(self,):
            return None, torch.ones(1)

    t = Test()
    assert torch.isclose(t._kl, torch.zeros(1))
    t()
    assert torch.isclose(t._kl, torch.ones(1))
