import torch

from gpytorch.likelihoods import GaussianLikelihood as BaseGL
from gpytorch.likelihoods import MultitaskGaussianLikelihood as BaseMGL

from gpytorch.lazy import LazyEvaluatedKernelTensor


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

    def marginal(self, function_dist, *params, **kwargs):
        r"""
        This overwrites the marginal() method in gpytorch._MultitaskGaussianLikelihoodBase.
        If 'noise' is None, this just calls the super() version of the method.
        Otherwise, it uses the _shaped_noise_covar() to add the noise to the diagonal of the covariance (see above).
        """

        if ('noise' in kwargs) and (kwargs['noise'] is not None):
            mean = function_dist.mean
            covar = self._shaped_noise_covar(mean.shape, noise=kwargs['noise'])
            
            return function_dist.__class__(mean, covar, interleaved=function_dist._interleaved)
        else:
            # Similar to the super().marginal() method, except no noise is added
            #     the overwrite of the _shaped_noise_covar (above), 
            #     already adds the noise to the diagonal, so the super().marginal() method would add it twice.
            mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

            # ensure that sumKroneckerLT is actually called
            if isinstance(covar, LazyEvaluatedKernelTensor):
                covar = covar.evaluate_kernel()
            
            return function_dist.__class__(mean, covar, interleaved=function_dist._interleaved)
        
        
        
