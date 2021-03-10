import attr
from torch import nn

from lantern.model.surface import Surface
from lantern.model.basis import Basis
from lantern.model.loss import Loss


@attr.s
class Model(nn.module):
    """The base model interface for *lantern*, learning a surface along a low-dimensional basis of mutational data.
    """

    basis: Basis = attr.ib()
    surface: Surface = attr.ib()

    @surface.validator
    def _surface_validator(self, attribute, value):
        if value.D != self.basis.D:
            raise ValueError(
                f"Basis ({self.basis.D}) and surface ({value.D}) do not have the same dimensionality."
            )

    def forward(self, X):

        Z = self.basis(X)
        f = self.surface(Z)

        return f
