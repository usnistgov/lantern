def _find_colinear(args):
    X, i = args
    for j in range(i):
        # easier to find co-linear this way b/c only 0 or 1
        if (X[:, i] == X[:, j]).all():
            return False
    return True

rule sir_fit:
    """Fit SIR model"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-fit.pkl"
    resources:
        mem = "32000M",
        time = "16:00:00"
    group: "figure"
    run:
        import pickle
        from itertools import combinations
        # from multiprocessing import Pool
        
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression
        from tqdm import tqdm

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
        print("data setup")
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # find co-linear columns
        print("find colinear")

        p = X.shape[1]
        ind = [i for i in range(p)]
        for i in range(p):
            for j in range(i + 1, p):
                # easier to find co-linear this way b/c only 0 or 1
                if (X[:, i] == X[:, j]).all() and j in ind:
                    ind.remove(j)
        Xtrim = X[:, ind]

        # setup SIR
        print("train SIR")
        sir = SlicedInverseRegression(n_directions=8, n_slices=30)

        sir.fit(Xtrim, y)
        X_sir = sir.transform(Xtrim)

        # save results
        print("save")
        res = {
            "model": sir,
            "X": X,
            "Xtrim": Xtrim,
            "y": y,
            "X_sir": X_sir,
            "ind": ind
        }

        with open(output[0], "wb") as of:
            pickle.dump(res, of)

rule sir_effects:
    """Compare sliced inverse regression directions to LANTERN effects"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "figures/{ds}-{phenotype}/{target}/sir-fit.pkl"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-effects.png"
    resources:
        mem = "32000M"
    group: "figure"
    run:
        import pickle

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

        # load fit
        with open(input[-1], "rb") as f:
            fit = pickle.load(f)
        sir = fit["model"]
        Xtrim = fit["Xtrim"]
        ind = fit["ind"]
        X_sir = fit["X_sir"]

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # remove ignored mutations
        W = W[ind, :]

        # setup figure
        K = fget(
            "zdim",
            default=K,
        )

        D = fget(
            "wdim",
            default=8,
        )

        alpha = 0.02
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

rule sir_effects_corr:
    """Correlation between sliced inverse regression directions to LANTERN effects"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "figures/{ds}-{phenotype}/{target}/sir-fit.pkl"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-effects-corr.png"
    resources:
        mem = "32000M"
    group: "figure"
    run:
        import pandas as pd
        import numpy as np                                    
        import matplotlib.pyplot as plt
        from sliced import SlicedInverseRegression
        from scipy.stats import pearsonr

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

        # load fit
        with open(input[-1], "rb") as f:
            fit = pickle.load(f)
        sir = fit["model"]
        Xtrim = fit["Xtrim"]
        ind = fit["ind"]
        X_sir = fit["X_sir"]

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # remove ignored mutations
        W = W[ind, :]

        # setup figure
        K = fget(
            "zdim",
            default=K,
        )

        D = fget(
            "wdim",
            default=8,
        )

        C = np.zeros((K, D))  # correlation
        S = np.zeros((K, D))  # significance

        for k in range(K):
            for d in range(D):
                # remove outliers
                _d = sir.directions_[d, :]
                _w = W[:, k]

                sel = abs(_d - np.mean(_d)) < 5 * np.std(_d)
                sel = sel & (abs(_w - np.mean(_w)) < 5 * np.std(_w))

                C[k, d], S[k, d] = pearsonr(sir.directions_[d, sel], W[sel, k])

        plt.imshow(C, aspect="auto", cmap="PRGn", origin="lower", vmin=-1, vmax=1)
        for k in range(K):
            for d in range(D):
                if S[k, d] < 0.05:
                    plt.text(
                        d - 0.25,
                        k - 0.5,
                        "*",
                        fontsize=84,
                        fontweight="heavy",
                        multialignment="center",
                    )

        plt.xticks(range(D), [f"$w_{d+1}$" for d in range(D)])
        plt.yticks(range(D), [f"$z_{d+1}$" for d in range(K)])
                    
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")


rule sir_dims:
    """Plot individual directions versus phenotype"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "figures/{ds}-{phenotype}/{target}/sir-fit.pkl"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-dims.png"
    resources:
        mem = "32000M"
    group: "figure"
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

        D = fget(
            "wdim",
            default=8,
        )

        # load fit
        with open(input[-1], "rb") as f:
            fit = pickle.load(f)
        sir = fit["model"]
        Xtrim = fit["Xtrim"]
        ind = fit["ind"]
        X_sir = fit["X_sir"]

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # remove ignored mutations
        W = W[ind, :]

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

        if raw is not None:
            y = y * df[raw].std() + df[raw].mean()

        plt.figure(figsize=(3*D, 3))
        for d in range(D):
            plt.subplot(1, D, d + 1)
            plt.hist2d(X_sir[:, d], y, bins=30, norm=mpl.colors.LogNorm())
            plt.xlabel(f"$w_{d+1}$")

            if d == 0:
                plt.ylabel(phenotype_name)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

rule sir_variance:
    """Compare sliced inverse regression eigen values to LANTERN variance"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "figures/{ds}-{phenotype}/{target}/sir-fit.pkl"
    output:
        "figures/{ds}-{phenotype}/{target}/sir-variance.png"
    resources:
        mem = "32000M"
    group: "figure"
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

        # load fit
        with open(input[-1], "rb") as f:
            fit = pickle.load(f)
        sir = fit["model"]
        Xtrim = fit["Xtrim"]
        ind = fit["ind"]
        X_sir = fit["X_sir"]

        # setup LANTERN
        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", K)
        W = model.basis.W_mu[:, model.basis.order].detach().numpy()
        X, y = ds[:len(ds)][:2]
        y = y[:, p].numpy()

        # remove ignored mutations
        W = W[ind, :]

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
