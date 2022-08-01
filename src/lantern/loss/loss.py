from typing import List

import attr
from torch import nn

from lantern import Module


class Loss(Module):
    """A loss component used in optimizing a model.
    """

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def loss(self, yhat, y, noise=None, *args, **kwargs) -> dict:
        raise NotImplementedError()


@attr.s(eq=False)
class Term(Loss):
    """A loss term used in optimizing a model.
    """

    def __add__(self, other):
        if isinstance(other, Composite):
            return Composite([self] + other.losses)
        elif isinstance(other, Term):
            return Composite([self] + [other])


@attr.s(eq=False)
class Composite(Loss):

    """The loss used to optimize a model, composed of individual Term's
    """

    losses: List[Term] = attr.ib()

    def __attrs_post_init__(self):
        self._losses = nn.ModuleList(self.losses)

    def loss(self, yhat, y, noise=None, *args, **kwargs):

        lss = {}
        for l in self.losses:
            d = l.loss(yhat, y, noise, *args, **kwargs)
            lss.update(d)

        # instead of storing this, just do it when it's time to call backward
        # lss["total"] = sum(lss.values())

        return lss

    def __add__(self, other):
        if isinstance(other, Composite):
            return Composite(self.losses + other.losses)
        elif isinstance(other, Term):
            return Composite(self.losses + [other])
