import attr
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean, Mean
from gpytorch.kernels import ScaleKernel, RQKernel, Kernel
import torch


from lantern.model.surface import Surface
from lantern.dataset import Dataset


@attr.s(cmp=False)
class Phenotype(ApproximateGP, Surface):
    """A phenotype surface, learned with an approximate GP.
    
    :param D: The phenotype dimension
    :type D: int
    :param K: The latent effect dimension
    :type K: int
    :param mean: The mean function of the GP
    :type mean: gpytorch.means.Mean
    :param kernel: The GP kernel function
    :type kernel: gpytorch.kernels.Kernel
    :param variational_strategy: The strategy for variational inference
    :type variational_strategy: gpytorch.variational.VariationalStrategy
    """

    D: int = attr.ib()
    dataset: Dataset = attr.ib()
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

    @property
    def Kbasis(self):
        return self.K

    def forward(self, z):
        """The forward prediction of the phenotype for a position in latent phenotype space.
        """
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

    @classmethod
    def fromDataset(cls, ds, *args, **kwargs):
        """Build a phenotype surface matching a dataset
        """

        return cls.build(ds.D, ds, *args, **kwargs)

    @classmethod
    def build(
        cls,
        D,
        ds,
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
        """Build a phenotype surface object.

        :param D: Number of dimensions of the (output) phenotype
        :type D: int
        :param K: Number of latent dimesions
        :type K: int
        :param Ni: Number of inducing points
        :type Ni: int, optional
        :param inducScale: Range to initialize inducing points over (uniform from [-inducScale, inducScale])
        :type inducScale: float, optional
        :param distribution: The distribution of the variational approximation
        :type distribution: gpytorch.VariationalDistribution
        :param mean: Mean function of the GP
        :type mean: gpytorch.means.Mean, optional
        :param kernel: The kernel of the GP
        :type kernel: gpytorch.kernels.Kernel, optional
        :param learn_inducing_locations: Whether to learn location of inducing points
        :type learn_inducing_locations: bool, optional
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

        return cls(D, ds, K, mean, kernel, strat, *args, **kwargs)
