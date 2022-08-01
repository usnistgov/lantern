import attr
import torch
from torch import nn
from torch.distributions import Gamma, Normal
from torch.distributions.kl import kl_divergence

from lantern.model import Variational
from lantern.model.basis import Basis


@attr.s(cmp=False)
class VariationalBasis(Basis, Variational):
    """A variational basis for reducing mutational data.
    """

    W_mu: nn.Parameter = attr.ib()
    W_log_sigma: nn.Parameter = attr.ib()
    log_alpha: nn.Parameter = attr.ib()
    log_beta: nn.Parameter = attr.ib()
    alpha_prior: Gamma = attr.ib()

    @classmethod
    def fromDataset(cls, ds, K, alpha_0=0.001, beta_0=0.001, meanEffectsInit=False):
        p = ds.p
        Wmu = torch.randn(p, K)

        # initialize first dimensions to mean effects
        if meanEffectsInit:
            mu = ds.meanEffects()
            Wmu[:, : mu.shape[1]] = mu

        Wmu = nn.Parameter(Wmu)

        # try to prevent some instabilities on gradients
        log_alpha = nn.Parameter(torch.randn(K))
        log_alpha.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
        log_beta = nn.Parameter(torch.randn(K))
        log_beta.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))

        return cls(
            Wmu,
            nn.Parameter(torch.randn(p, K) - 3),
            log_alpha,
            log_beta,
            Gamma(alpha_0, beta_0),
        )

    @classmethod
    def build(cls, p, K, alpha_0=0.001, beta_0=0.001):
        Wmu = torch.randn(p, K)

        Wmu = nn.Parameter(Wmu)

        # try to prevent some instabilities on gradients
        log_alpha = nn.Parameter(torch.randn(K))
        log_alpha.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
        log_beta = nn.Parameter(torch.randn(K))
        log_beta.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))

        return cls(
            Wmu,
            nn.Parameter(torch.randn(p, K) - 3),
            log_alpha,
            log_beta,
            Gamma(alpha_0, beta_0),
        )

    @property
    def p(self):
        return self.W_mu.shape[0]

    @property
    def K(self):
        return self.W_mu.shape[1]

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
