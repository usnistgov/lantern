from torch import nn
import torch


class Variational(nn.Module):
    def __init__(self):
        super(Variational, self).__init__()

        self.register_buffer("_kl", torch.zeros(1))

    def forward(self, *args, **kwargs):

        fwd, kl = self._forward(*args, **kwargs)
        self._kl = kl
        return fwd

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()
