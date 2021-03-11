import attr
import torch
from torch import nn
from torch.distributions import Gamma, Normal
from torch.distributions.kl import kl_divergence

from lantern.model import Variational
from lantern.model.basis import Basis


def kl_mvn_diag(m1, s1, m2, s2):
    kl = (
        (s2.prod() / s1.prod()).log()
        - s1.shape[0]
        + (s1 / s2).sum()
        + torch.mv((1 / s2).diag(), (m1 - m2)).dot(m1 - m2)
    )
    kl *= 0.5
    return kl


@attr.s()
class VariationalBasis(Basis, Variational):
    """A variational basis for reducing mutational data.
    """

    alpha_0: float = attr.ib(default=0.001)
    beta_0: float = attr.ib(default=0.001)

    def __attrs_post_init__(self):
        super(VariationalBasis, self).__attrs_post_init__()

        self.W_mu = nn.Parameter(torch.randn(self.p, self.D))
        self.W_log_sigma = nn.Parameter(torch.randn(self.p, self.D) - 3)

        self.alpha_prior = Gamma(self.alpha_0, self.beta_0)
        self.log_alpha = nn.Parameter(torch.randn(self.D))
        self.log_beta = nn.Parameter(torch.randn(self.D))

    def qalpha(self, detach=False):
        if detach:
            return Gamma(self.log_alpha.exp().detach(), self.log_beta.exp().detach())
        return Gamma(self.log_alpha.exp(), self.log_beta.exp())

    def kl_loss(self):
        # variational approximations
        # qalpha = Gamma(self.log_alpha.exp(), self.log_beta.exp())
        qalpha = self.qalpha()
        qW = Normal(self.W_mu, self.W_log_sigma.exp())

        # samples
        if self.training:
            alpha = qalpha.rsample()
            W = qW.rsample()
        else:
            alpha = qalpha.mean
            W = qW.mean

        # prior loss
        K, L = W.shape

        wprior = None
        klW = None
        scale = (1 / alpha).sqrt().repeat(K, 1)
        wprior = Normal(0, scale)
        klW = kl_divergence(qW, wprior)
        kla = kl_divergence(qalpha, self.alpha_prior)

        return klW, kla, W, alpha

    def _forward(self, x):

        klW, kla, W, alpha = self.kl_loss()
        loss = klW.sum() + kla.sum()

        # embed
        z = torch.matmul(x, W)

        return z, loss

    @property
    def order(self):

        gamma = self.qalpha(detach=True)
        srt = gamma.mean.sort().indices
        return srt

    def loss(self, N, *args, **kwargs):
        from lantern.loss import KL

        return KL("variational_basis", self, N)
