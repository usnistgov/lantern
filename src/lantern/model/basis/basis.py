import attr
import torch


@attr.s()
class Basis(torch.nn.Module):
    """A dimension reducing basis for mutational data.
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
