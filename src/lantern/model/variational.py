from torch import nn
import torch


class Variational(nn.Module):
    """A Variational module, that keeps track of it's running KL loss.
    """

    def __init__(self):
        super(Variational, self).__init__()

        self.register_buffer("_kl", torch.tensor(0.0))

    def forward(self, *args, **kwargs):

        fwd, kl = self._forward(*args, **kwargs)
        self._kl = kl
        return fwd

    def _forward(self, *args, **kwargs):
        """The base forward method for variational objects, returning a tuple of actual forward op and kl loss
        """
        raise NotImplementedError()
