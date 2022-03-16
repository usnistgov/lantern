
rule simulate_landscape:
    """Simulate a landscape from the LANTERN prior"""
    output:
        "data/processed/sim{name}.csv",
        "data/processed/sim{name}-w.csv",
        "figures/sim{name}-phenotype/surface.png",
        "figures/sim{name}-phenotype/surface-crossplot.png",
    resources:
        mem_mb = "32000M",
    run:
        import pandas as pd
        from scipy.stats import gamma, binom
        import torch
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.kernels import RQKernel
        from numpy.polynomial import Hermite

        from lantern.dataset import Tokenizer
        from lantern.model import Model
        from lantern.model.basis import VariationalBasis
        from lantern.model.surface import Phenotype

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"simulations/{wildcards.name}/{pth}", default=default)

        K = dsget("K", None)
        p = dsget("p", None)
        N = dsget("N", None)
        mrate = dsget("mutation_rate", None)
        sigmay = dsget("sigma_y", None)

        tokens = [str(n).zfill(int(np.floor(np.log10(p))) + 1) for n in range(p)]
        assert len(set(tokens)) == p

        # simulate variants
        variants = [""]  # alway see wildtype
        for n in range(N - 1):

            # add one to avoid a bunch of WT
            nsub = binom(p - 1, mrate).rvs() + 1
            subs = np.random.choice(tokens, nsub, replace=False)
            variants.append(":".join(subs))

        tokenizer = Tokenizer.fromVariants(variants)
        X = tokenizer.tokenize(*variants)

        # simulate effects
        # generate a sequence of variances with expected decay
        # eta = 5
        # pg = gamma(eta, scale=1/eta)
        # sigma = pg.rvs(K).cumprod()
        # W = (torch.randn(p, K) * torch.tensor(sigma[None, :])).float()
        W = torch.randn(p, K)

        # simulate data
        Z = torch.mm(X, W)

        # Gaussian surface 
        # f = (-Z.norm(dim=1)/2).exp()

        # add Hermite polynomials, for each basis, normalize to 1
        f = torch.zeros(N)
        fks = []
        for k in range(K):
            fk = torch.from_numpy(
                Hermite.basis(2*k + 1)((Z[:, k] / Z[:, k].abs().max()).numpy())
            )
            fk = fk/fk.abs().max()

            print([torch.dot(fk / fk.norm(), _fk / _fk.norm()).item() for _fk in fks])
            fks.append(fk)

            f = f + fk

        # probably want to do this
        # f = (f - f.mean()) / f.std()

        plt.figure(figsize=(1.5*K, 1.5*K))
        for i, fi in enumerate(fks):
            for j, fj in enumerate(fks):
                plt.subplot(K, K, K*j + i + 1)
                plt.hist2d(fi.numpy(), fj.numpy(), bins=30, norm=mpl.colors.LogNorm())
        plt.tight_layout()
        plt.savefig(output[3], bbox_inches="tight")
        
        # GP
        # kernel = RQKernel()
        # kernel.lengthscale = 2.0
        # kernel.alpha = 1

        # Kz = kernel(Z)
        # Ky = Kz + torch.eye(N) * sigmay
        # print("mvn")

        # y = MultivariateNormal(torch.zeros(N), Ky).sample()
        # y = MultivariateNormal(f, Ky).sample()
        
        y = torch.distributions.Normal(f, sigmay).sample()

        # build the dataset
        df = pd.DataFrame({"substitutions": variants, "phenotype": y.numpy()})
        df.to_csv(output[0], index=False)

        pd.DataFrame(
            W.numpy(), columns=[f"z{z+1}" for z in range(K)], index=tokens
        ).to_csv(output[1], index=False)

        if K == 1:
            plt.scatter(Z[:, 0], y, alpha=0.6)
        else:
            plt.scatter(Z[:, 0], Z[:, 1], c=y)
        plt.tight_layout()
        plt.savefig(output[2])
