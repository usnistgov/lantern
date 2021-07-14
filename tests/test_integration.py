import pandas as pd
from torch.optim import Adam

from lantern.dataset import Dataset
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model.likelihood import GaussianLikelihood


def test_quickstart():
    """Test the quickstart from the docs front-page
    """

    # create a dataframe containing GPL data
    df = pd.DataFrame(
        {"substitutions": ["", "+a", "+b", "+a:+b"], "phenotype": [0.0, 1.0, 1.0, 0.8]},
    )

    # convert the data to a LANTERN dataset
    ds = Dataset(df)

    # build a LANTERN model based on the dataset, using an upper-bound
    # of K latent dimensions
    model = Model(
        VariationalBasis.fromDataset(ds, 3),
        Phenotype.fromDataset(ds, 3, Ni=50),
        GaussianLikelihood(),
    )

    loss = model.loss(N=len(ds))
    X, y = ds[: len(ds)]

    optimizer = Adam(loss.parameters(), lr=0.1)

    # get initial loss
    yhat = model(X)
    lss = loss(yhat, y)
    baseline = sum(lss.values())

    for i in range(20):
        optimizer.zero_grad()
        yhat = model(X)
        lss = loss(yhat, y)
        total = sum(lss.values())
        total.backward()
        optimizer.step()

    # loss should have decreased, occasionally fails stochastically
    assert total < baseline
