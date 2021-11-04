
rule simulate_landscape:
    """Simulate a landscape from the LANTERN prior"""
    output:
        "data/processed/sim{name}.csv",
        "data/processed/sim{name}-w.csv",
        "figures/sim{name}-phenotype/surface.png",
    resources:
        mem_mb = "32000M",
    run:
        import pandas as pd
        from scipy.stats import gamma, binom
        import torch
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.kernels import RQKernel

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

        tokens = [str(n).zfill(int(np.floor(np.log10(p)))) for n in range(p)]
        assert len(set(tokens)) == p

        # simulate effects
        # generate a sequence of variances with expected decay
        # eta = 5
        # pg = gamma(eta, scale=1/eta)
        # sigma = pg.rvs(K).cumprod()
        # W = (torch.randn(p, K) * torch.tensor(sigma[None, :])).float()
        W = torch.randn(p, K) * 1

        # simulate variants
        variants = [""]  # alway see wildtype
        for n in range(N - 1):

            # add one to avoid a bunch of WT
            nsub = binom(p - 1, mrate).rvs() + 1
            subs = np.random.choice(tokens, nsub, replace=False)
            variants.append(":".join(subs))

        tokenizer = Tokenizer.fromVariants(variants)
        X = tokenizer.tokenize(*variants)

        # simulate data
        Z = torch.mm(X, W)

        # Gaussian surface 
        f = (-Z.norm(dim=1)/2).exp()

        # GP
        kernel = RQKernel()
        kernel.lengthscale = 2.0
        kernel.alpha = 1

        Kz = kernel(Z)
        Ky = Kz + torch.eye(N) * sigmay
        print("mvn")

        # y = MultivariateNormal(torch.zeros(N), Ky).sample()
        y = MultivariateNormal(f, Ky).sample()

        print("done")

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
