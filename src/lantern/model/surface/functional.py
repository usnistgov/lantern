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


@attr.s(cmp=False)
class Functional(ApproximateGP, Surface):

    Z: torch.tensor = attr.ib()
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

    @property
    def M(self):
        """Number of covariates
        """
        return self.Z.shape[1]

    @property
    def n(self):
        """Number of function observations
        """
        return self.Z.shape[0]

    @property
    def Kbasis(self):
        return self.K - self.M

    @staticmethod
    def _expand(*tensors):
        """Make a cartesian product of tensors.
        """

        # make cartesian product of tensor indices. each row of prod
        # stores the row index of each tensor to take for a given
        # observation. each column of prod is for one of the input tensors
        ind = [torch.arange(t.shape[0]) for t in tensors]
        prod = torch.cartesian_prod(*ind)

        # build cartesian product tensor of individual tensor elements
        offset = 0
        ret = torch.zeros(prod.shape[0], sum([t.shape[1] for t in tensors]))
        if tensors[0].is_cuda:
            ret = ret.cuda()
        for i, t in enumerate(tensors):
            ret[:, offset : offset + t.shape[1]] = t[prod[:, i], :]
            offset += t.shape[1]

        return ret

    def __call__(self, z):
        """Override the gpytorch call, creating the right size and shape tensor from the input
        """
        z = self._expand(z, self.Z)
        return super(Functional, self).__call__(z)

    def forward(self, z):
        """The forward prediction of the functional phenotype for a position in latent phenotype space.
        """
        mean_x = self.mean(z)
        covar_x = self.kernel(z)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def fromDataset(cls, Z, ds, *args, **kwargs):
        """Build a functional phenotype surface matching a dataset
        """

        return cls.build(Z, ds.D, *args, **kwargs)

    @classmethod
    def build(
        cls,
        Z,
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
        """Build a functional phenotype surface object.

        :param Z: the fixed functional values of the phenotype
        :type Z: torch.tensor
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

        # number of fixed dims plus provided K
        Ktotal = Z.shape[1] + K

        if D > 1:
            shape = (D, Ni, Ktotal)
        else:
            shape = (Ni, Ktotal)

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
                k1 = RQKernel(
                    ard_num_dims=K, active_dims=range(K), batch_shape=torch.Size([D])
                )
                k2 = RQKernel(
                    ard_num_dims=Z.shape[1],
                    active_dims=range(K, Ktotal),
                    batch_shape=torch.Size([D]),
                )
            else:
                k1 = RQKernel(ard_num_dims=K, active_dims=range(K))
                k2 = RQKernel(ard_num_dims=Z.shape[1], active_dims=range(K, Ktotal))

            # no lengthscale for basis dimensions
            if k1.has_lengthscale:
                k1.raw_lengthscale.requires_grad = False

            kernel = k1 + k2

            # scale component
            if D > 1:
                kernel = ScaleKernel(kernel, batch_shape=torch.Size([D]))
            else:
                kernel = ScaleKernel(kernel)

        return cls(Z, D, Ktotal, mean, kernel, strat, *args, **kwargs)
