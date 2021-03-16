import attr

from lantern.loss import Term
from lantern.model import Variational


@attr.s(eq=False)
class KL(Term):

    """ A variational KL loss term.
    """

    name: str = attr.ib()
    component: Variational = attr.ib(repr=False)
    N: int = attr.ib(repr=False)

    def loss(self, *args, **kwargs):

        return {self.name: self.component._kl / self.N}
