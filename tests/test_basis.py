import torch
import pandas as pd

from lantern.dataset import Dataset
from lantern.model.basis import Basis


def test_order():

    p = 200
    K = 10

    b = Basis(p=p, K=K)
    assert torch.allclose(b.order, torch.arange(K))


def test_build():
    df = pd.DataFrame(
        {"substitutions": ["a1b", "c2d"], "phenotype": [0.0, 1.0], "error": [0.1, 0.2],}
    )
    ds = Dataset(df)
    b = Basis.fromDataset(ds, 10)

    assert b.p == 2
    assert b.K == 10
