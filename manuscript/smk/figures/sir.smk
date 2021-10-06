rule sir_compare:
    """Compare sliced inverse regression to LANTERN"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-effects.png"
    run:
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression

        from lantern.dataset.tokenizer import Tokenizer

        # setup SIR
        data = pd.read_csv("data/processed/gfp.csv")
        tok = Tokenizer.fromVariants(data.substitutions.replace(np.nan, ""))

        X = tok.tokenize(*data.substitutions.replace(np.nan, "")).numpy()
        y = data.phenotype

        sir = SlicedInverseRegression(n_directions=8, n_slices=30)

        sir.fit(X, y)

        # setup LANTERN
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        K = dsget("K", 8)

        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()

        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/sir/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
                default=default,
            )

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


