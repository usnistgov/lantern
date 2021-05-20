import torch
from torch import nn
from torch.nn import functional as F


class Feedforward(nn.Module):
    def __init__(self, p, K, D, depth=1, width=32):
        super(Feedforward, self).__init__()

        # always need at least these many layers
        layers = [
            nn.Linear(p, K),
            nn.ReLU(),
            nn.Linear(K, width),
            nn.ReLU(),
        ]

        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, D))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):

        for l in self.layers:
            x = l(x)

        return x

    def loss(self, x, y, noise=None):

        yhat = self(x)
        loss = F.mse_loss(yhat, y.float(), reduction="none")

        if noise is not None:
            nz = torch.count_nonzero(noise, dim=0)

            # check that noise is not all zero
            for i, nnz in enumerate(nz):
                if nnz > 0:
                    loss[:, i] = loss[:, i] / noise[:, i]

        return loss.mean()
