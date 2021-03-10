import attr
import torch


@attr.s()
class Basis(torch.nn.Module):
    """A dimension reducing basis for mutational data.

    :param p: Input dimension, e.g. the number of mutations
    :type p: int
    :param D: output dimension, e.g. the number of latent directions
    :type D: int
    """

    p: int = attr.ib()
    D: int = attr.ib()

    def __attrs_post_init__(self):
        super(Basis, self).__init__()

    @property
    def order(self):
        """The rank order of latent dimensions
        """
        return torch.arange(self.D)

    @classmethod
    def fromDataset(
        cls, ds,
    ):
        return cls(ds.p, ds.D)
