import torch
import pandas as pd
import pytest

from lantern.dataset import Dataset
from lantern.model.basis import Basis


def test_order():

    K = 10

    class DummyBasis(Basis):
        @property
        def K(self):
            return K

    b = DummyBasis()
    assert torch.allclose(b.order, torch.arange(K))


def test_build():
    df = pd.DataFrame(
        {"substitutions": ["a1b", "c2d"], "phenotype": [0.0, 1.0], "error": [0.1, 0.2],}
    )
    ds = Dataset(df)

    with pytest.raises(NotImplementedError):
        Basis.fromDataset(ds, 10)
