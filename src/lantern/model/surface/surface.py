import attr

from lantern.module import Module


@attr.s
class Surface(Module):

    D: int = attr.ib()

    def __attrs_post_init__(self):
        super(Surface, self).__init__()
