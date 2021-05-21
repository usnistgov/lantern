from gpytorch.likelihoods import GaussianLikelihood as BaseGL
from gpytorch.likelihoods import MultitaskGaussianLikelihood as BaseMGL


class GaussianLikelihood(BaseGL):
    """A modification to the base class:`gpytorch.likelihoods.GaussianLikelihood` that supports combined base noise with provided noise.
    """

    def _shaped_noise_covar(self, base_shape, noise=None):
        noise_covar = super()._shaped_noise_covar(base_shape)
        if noise is not None:
            return noise_covar.add_diag(noise)
        return noise_covar


class MultitaskGaussianLikelihood(BaseMGL):
    """A modification to the base class:`gpytorch.likelihoods.MultitaskGaussianLikelihood` that supports combined base noise with provided noise.
    """

    def _shaped_noise_covar(self, base_shape, noise=None):
        noise_covar = super()._shaped_noise_covar(base_shape)
        if noise is not None:
            return noise_covar.add_diag(noise)
        return noise_covar
