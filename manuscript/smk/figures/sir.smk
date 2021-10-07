rule sir_effects:
    """Compare sliced inverse regression directions to LANTERN effects"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-effects.png"
    resources:
        mem = "32000M"
    run:
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression

        from lantern.dataset.tokenizer import Tokenizer

        # Configuration
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        K = dsget("K", 8)

        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
                default=default,
            )

        p = fget(
            "p",
            default=0,
        )

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # setup SIR
        sir = SlicedInverseRegression(n_directions=8, n_slices=30)

        sir.fit(X, y)

        # setup figure
        K = fget(
            "zdim",
            default=K,
        )

        D = fget(
            "wdim",
            default=8,
        )

        alpha = 0.01
        plt.figure(figsize=(3 * D, 3 * K))
        for k in range(K):
            for d in range(D):
                plt.subplot(K, D, k * D + d + 1)
                plt.hist2d(
                    sir.directions_[d, :],
                    W[:, k],
                    norm=mpl.colors.LogNorm(),
                    bins=(
                        np.linspace(
                            np.quantile(sir.directions_[d, :], alpha / 2),
                            np.quantile(sir.directions_[d, :], 1 - alpha / 2),
                            30,
                        ),
                        np.linspace(
                            np.quantile(W[:, k], alpha / 2),
                            np.quantile(W[:, k], 1 - alpha / 2),
                            30,
                        ),
                    ),
                )

                if k == K - 1:
                    plt.xlabel(f"$w_{d+1}$")
                if d == 0:
                    plt.ylabel(f"$z_{k+1}$")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")


rule sir_dims:
    """Plot individual directions versus phenotype"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-dims.png"
    resources:
        mem = "32000M"
    run:
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression

        from lantern.dataset.tokenizer import Tokenizer

        # Configuration
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        K = dsget("K", 8)

        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
                default=default,
            )

        p = fget(
            "p",
            default=0,
        )

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        phenotype_name = get(
            config,
            f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/phenotype_name",
            default="",
        )
        raw = get(
            config,
            f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/raw",
            default=None,
        )

        # setup SIR
        sir = SlicedInverseRegression(n_directions=8, n_slices=30)

        sir.fit(X, y)
        X_sir = sir.transform(X)

        if raw is not None:
            y = y * df[raw].std() + df[raw].mean()

        plt.figure(figsize=(8, 6))
        for d in range(8):
            plt.subplot(2, 4, d + 1)
            plt.hist2d(X_sir[:, d], y, bins=30, norm=mpl.colors.LogNorm())
            plt.xlabel(f"$w_{d+1}$")

            if d % 4 == 0:
                plt.ylabel(phenotype_name)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

rule sir_variance:
    """Compare sliced inverse regression eigen values to LANTERN variance"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-variance.png"
    resources:
        mem = "32000M"
    run:
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression

        from lantern.dataset.tokenizer import Tokenizer

        # Configuration
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        K = dsget("K", 8)

        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
                default=default,
            )

        p = fget(
            "p",
            default=0,
        )

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()


        # setup SIR
        sir = SlicedInverseRegression(n_directions=8, n_slices=30)

        sir.fit(X, y)

        _, ax = plt.subplots()
        
        plt.plot(np.arange(1, 9),sir.eigenvalues_, marker="o")
        ax.spines['left'].set_color('C0')
        ax.tick_params(axis='y', colors='C0')
        ax.yaxis.label.set_color("C0")
        plt.ylabel("SIR eigen values")
        plt.semilogy()

        plt.xlabel("dimension")

        ax = plt.twinx()

        with torch.no_grad():
            mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
            mu = mu[model.basis.order]

        plt.plot(np.arange(1, 9), mu, marker="o", c="C1")

        ax.spines['right'].set_color('C1')
        ax.tick_params(axis='y', colors='C1')
        ax.yaxis.label.set_color("C1")
        ax.set_ylabel('LANTERN $\sigma^2_k$', rotation=-90, labelpad=20)
        plt.semilogy()

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
