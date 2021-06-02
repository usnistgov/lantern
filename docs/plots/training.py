import pandas as pd
from torch.optim import Adam
import torch
import matplotlib.pyplot as plt

from lantern.dataset import Dataset
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype

# create a dataframe containing GPL data
df = pd.read_csv("../example.csv")

# convert the data to a LANTERN dataset
ds = Dataset(df)

# build a LANTERN model based on the dataset, using an upper-bound
# of 8 latent dimensions
model = Model(
    VariationalBasis.fromDataset(ds, 8, meanEffectsInit=True),
    Phenotype.fromDataset(ds, 8),
)

loss = model.loss(N=len(ds))
X, y = ds[: len(ds)]

optimizer = Adam(loss.parameters(), lr=0.01)
hist = []
for i in range(100):
    optimizer.zero_grad()
    yhat = model(X)
    lss = loss(yhat, y)
    total = sum(lss.values())
    total.backward()
    optimizer.step()
    hist.append(total.item())

plt.plot(hist)
plt.xlabel("epoch")
plt.ylabel("loss")

torch.save(model.state_dict(), "model.pt")
