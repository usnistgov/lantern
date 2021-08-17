import torch

from gpytorch.likelihoods import GaussianLikelihood as BaseGL
from gpytorch.likelihoods import MultitaskGaussianLikelihood as BaseMGL


class GaussianLikelihood(BaseGL):
    """A modification to the base class:`gpytorch.likelihoods.GaussianLikelihood` that supports combined base noise with provided noise.
    """

    def _shaped_noise_covar(self, base_shape, noise=None, *args, **kwargs):
        noise_covar = super()._shaped_noise_covar(base_shape)
        if noise is not None:
            return noise_covar.add_diag(noise)
        return noise_covar


class MultitaskGaussianLikelihood(BaseMGL):
    """A modification to the base class:`gpytorch.likelihoods.MultitaskGaussianLikelihood` that supports combined base noise with provided noise.
    """

    def __init__(
        self,
        num_tasks,
        rank=0,
        task_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=False,  # change this default
        has_task_noise=True,
    ):
        super(MultitaskGaussianLikelihood, self).__init__(
            num_tasks,
            rank=rank,
            task_prior=task_prior,
            batch_shape=batch_shape,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            has_global_noise=has_global_noise,
            has_task_noise=has_task_noise,
        )

    def _shaped_noise_covar(self, base_shape, noise=None, *args, **kwargs):
        noise_covar = super()._shaped_noise_covar(base_shape)
        if noise is not None:
            # reshape to match diagonal shape in multitask case
            return noise_covar.add_diag(noise.reshape(noise.shape[0] * noise.shape[1]))
        return noise_covar
