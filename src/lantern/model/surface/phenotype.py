from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import VariationalDistribuiton
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.means import ConstantMean, Mean
from gpytorch.kernel import Kernel, ScaleKernel, RQKernel
import torch
import attr

from lantern.model.surface import Surface


@attr.s
class Phenotype(ApproximateGP, Surface):
    """A phenotype surface, learned with an approximate GP.
    """

    inducing: torch.Tensor = attr.ib()
    kernel: Kernel = attr.ib(default=ScaleKernel(RQKernel()))
    mean: Mean = attr.ib(default=ConstantMean)
    distribution: VariationalDistribuiton = attr.ib(
        default=CholeskyVariationalDistribution
    )
    learn_inducing_locations: bool = attr.ib(default=True)

    def __attrs_post_init__(self):

        size = torch.Size([])
        if self.D > 1:
            size = torch.Size([self.D])

        variational_distribution = self.distribution(
            self.inducing.size(-2), batch_shape=size
        )
        variational_strategy = VariationalStrategy(
            self,
            self.inducing,
            variational_distribution,
            learn_inducing_locations=self.learn_inducing_locations,
        )

        if self.D > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=self.D
            )

        ApproximateGP.__init__(self, variational_strategy)

        self.mean_module = self.mean(batch_shape=size)

    def forward(self, z):
        mean_x = self.mean_module(z)
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
