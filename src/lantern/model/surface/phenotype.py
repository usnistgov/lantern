import attr
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean, Mean
from gpytorch.kernels import ScaleKernel, RQKernel, Kernel
import torch


from lantern import Module


@attr.s(cmp=False)
class Phenotype(ApproximateGP, Module):
    """A phenotype surface, learned with an approximate GP.
    """

    D: int = attr.ib()
    K: int = attr.ib()

    mean: Mean = attr.ib()
    kernel: Kernel = attr.ib()

    variational_strategy: VariationalStrategy = attr.ib()

    def __attrs_post_init__(self):

        # hack to deal with circular inits
        if self.D == 1:
            # self.variational_strategy.model = self
            object.__setattr__(self.variational_strategy, "model", self)
        else:
            # self.variational_strategy.base_variational_strategy.model = self
            object.__setattr__(
                self.variational_strategy.base_variational_strategy, "model", self
            )

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
    def fromDataset(cls, ds, *args, **kwargs):

        return cls.build(ds.D, *args, **kwargs)

    @classmethod
    def build(
        cls,
        D,
        K,
        Ni=800,
        inducScale=10,
        distribution=CholeskyVariationalDistribution,
        mean=None,
        kernel=None,
        learn_inducing_locations=True,
        *args,
        **kwargs
    ):
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
        if D > 1:
            shape = (D, Ni, K)
        else:
            shape = (Ni, K)

        inducing_points = -inducScale + 2 * inducScale * torch.rand(*shape)

        size = torch.Size([])
        if D > 1:
            size = torch.Size([D])

            if len(inducing_points.shape) != 3:
                raise ValueError("Should have D x I x K inducing points!")

        variational_distribution = distribution(
            inducing_points.size(-2), batch_shape=size
        )
        strat = VariationalStrategy(
            None,  # this shouldn't actually be needed and will get filled in after init
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        if D > 1:
            strat = IndependentMultitaskVariationalStrategy(strat, num_tasks=D)

        if mean is None:
            mean = ConstantMean(batch_shape=size)
        if kernel is None:
            # rq component
            if D > 1:
                kernel = RQKernel(ard_num_dims=K, batch_shape=torch.Size([D]))
            else:
                kernel = RQKernel(ard_num_dims=K)
            if kernel.has_lengthscale:
                kernel.raw_lengthscale.requires_grad = False

            # scale component
            if D > 1:
                kernel = ScaleKernel(kernel, batch_shape=torch.Size([D]))
            else:
                kernel = ScaleKernel(kernel)

        return cls(D, K, mean, kernel, strat, *args, **kwargs)
