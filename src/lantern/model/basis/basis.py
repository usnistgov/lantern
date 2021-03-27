import attr
import torch

from lantern import Module


@attr.s(cmp=False)
class Basis(Module):
    """A dimension reducing basis for mutational data.

    :param p: Input dimension, e.g. the number of mutations
    :type p: int
    :param K: output dimension, e.g. the number of latent directions
    :type K: int
    """

    # p: int = attr.ib()
    # K: int = attr.ib()

    # def __attrs_post_init__(self):
    #     super(Basis, self).__init__()

    @property
    def p(self):
        raise NotImplementedError()

    @property
    def K(self):
        raise NotImplementedError()

    @property
    def order(self):
        """The rank order of latent dimensions
        """
        return torch.arange(self.K)

    @classmethod
    def fromDataset(cls, ds, K):
        raise NotImplementedError()
