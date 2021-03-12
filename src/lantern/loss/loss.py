from typing import List

import attr
from torch import nn

from lantern import Module


@attr.s
class Term(Module):
    """A loss term used in optimizing a model.
    """

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def loss(self, yhat, y, noise=None, *args, **kwargs) -> dict:
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, Loss):
            return Loss([self] + other.losses)
        elif isinstance(other, Term):
            return Loss([self] + [other])


@attr.s
class Loss(Module):

    """The loss used to optimize a model, composed of individual Term's
    """

    losses: List[Term] = attr.ib()

    def loss(self, yhat, y, noise=None, *args, **kwargs):

        lss = {}
        for l in self.losses:
            lss.update(l.loss(yhat, y, noise, *args, **kwargs))

        # instead of storing this, just do it when it's time to call backward
        # lss["total"] = sum(lss.values())

        return lss

    def __add__(self, other):
        if isinstance(other, Loss):
            return Loss(self.losses + other.losses)
        elif isinstance(other, Term):
            return Loss(self.losses + [other])
