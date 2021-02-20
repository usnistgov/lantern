from typing import List

import attr
from torch import nn


class Term(nn.Module):
    """A loss term used in optimizing a model.
    """

    def loss(self, yhat, y, noise, *args, **kwargs) -> dict:
        raise NotImplementedError()


@attr.s
class Loss:

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
