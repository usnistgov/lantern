from lantern.module import Module


class Surface(Module):
    @property
    def Kbasis(self):
        """The number of dimensions provided by the basis
        """
        raise NotImplementedError()
