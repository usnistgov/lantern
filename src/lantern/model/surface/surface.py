from torch import nn
import attr


@attr.s
class Surface(nn.Module):

    D: int = attr.ib()

    def __attrs_post_init__(self):
        super(Surface, self).__init__()
