from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RQKernel
import torch


class Phenotype(ApproximateGP):
    """A phenotype surface, learned with an approximate GP.
    """

    def __init__(
        self,
        D,
        inducing_points,
        strategy=CholeskyVariationalDistribution,
        mean=ConstantMean,
        learn_inducing_locations=True,
    ):
        size = torch.Size([])
        if D > 1:
            size = torch.Size([D])

            if len(inducing_points.shape) != 3:
                raise ValueError("Should have D x I x K inducing points!")

        variational_distribution = strategy(inducing_points.size(-2), batch_shape=size)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        if D > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=D
            )

        super(Phenotype, self).__init__(variational_strategy)

        self.D = D
        self.K = inducing_points.size(-1)
        self.mean = mean(batch_shape=size)

        # rq component
        if self.D > 1:
            kernel = RQKernel(ard_num_dims=self.K, batch_shape=torch.Size([self.D]))
        else:
            kernel = RQKernel(ard_num_dims=self.K)
        if kernel.has_lengthscale:
            kernel.raw_lengthscale.requires_grad = False

        # scale component
        if self.D > 1:
            kernel = ScaleKernel(kernel, batch_shape=torch.Size([self.D]))
        else:
            kernel = ScaleKernel(kernel)

        self.kernel = kernel

    def forward(self, z):
        mean_x = self.mean(z)
        covar_x = self.kernel(z)
        return MultivariateNormal(mean_x, covar_x)

    def _get_induc(self):
        strat = self.variational_strategy
        if isinstance(strat, IndependentMultitaskVariationalStrategy):
            strat = strat.base_variational_strategy

        return strat.inducing_points.detach().cpu().numpy()

    def _set_induc(self, induc):
        strat = self.variational_strategy
        if isinstance(strat, IndependentMultitaskVariationalStrategy):
            strat = strat.base_variational_strategy

        device = strat.inducing_points.device
        strat.inducing_points.data[:] = torch.from_numpy(induc).to(device)

    def loss(self, *args, **kwargs):
        from lantern.loss import ELBO_GP

        return ELBO_GP.fromGP(self, *args, **kwargs)

    @classmethod
    def fromDataset(cls, ds, K, Ni=800, inducScale=10, *args, **kwargs):
        """Build a phenotype surface from a dataset.

        :param ds: Dataset for build a phenotype from.
        :type ds: lantern.dataset.Dataset
        :param K: Number of latent dimesions
        :type K: int
        :param Ni: Number of inducing points
        :type Ni: int
        :param inducScale: Range to initialize inducing points over (uniform from [-inducScale, inducScale])
        :type inducScale: float
        """
        D = ds.D
        if D > 1:
            shape = (D, Ni, K)
        else:
            shape = (Ni, K)

        return cls(
            D, -inducScale + 2 * inducScale * torch.rand(*shape), *args, **kwargs
        )
