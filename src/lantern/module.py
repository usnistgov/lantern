from torch import nn
import attr


@attr.s(cmp=False)
class Module(nn.Module):
    """A base module for lantern components

    This module is necessary to play nicely b/w attrs and
    pytorch. Some discussion is available here:
    https://github.com/python-attrs/attrs/issues/393#issuecomment-510148031
    """

    def __attrs_pre_init__(self):
        # torch module is initialized before assigning attributes
        nn.Module.__init__(self)
