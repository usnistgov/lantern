from gpytorch.variational import IndependentMultitaskVariationalStrategy

rule gradient:
    """
    Gradient plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/gradient.png"

    run:
        from lantern.diffops import grad
        from lantern.diffops import metric

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/diffops/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
                default=default,
            )

        # surface parameters
        raw = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/raw",
            default=None,
        )
        log = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/log",
            default=False,
        )
        p = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/p",
            default=0,
        )
        alpha = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/alpha",
            default=0.01,
        )
        cbar_title = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_title",
            default=None,
        )
        cbar_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_kwargs",
            default={},
        )
        
        # figure parameters
        N = fget(
            "N",
            default=50,
        )
        fig_kwargs = fget(
            "fig_kwargs",
            default=dict(figsize=(5, 3), dpi=150),
        )
        zlim = fget(
            "zlim",
            default=None,
        )
        plot_kwargs = fget(
            "plot_kwargs",
            default={},
        )
        cbar_title = fget("cbar_title", default=cbar_title)

        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", dsget("K", 8))
        model.eval()

        d0, d1 = model.basis.order[:2]

        strat = model.surface.variational_strategy
        if isinstance(strat, IndependentMultitaskVariationalStrategy):
            strat = strat.base_variational_strategy
        z0 = strat.inducing_points.detach()

        if z0.ndim > 2:
            z0 = z0[p, :, :]

        # dataset robustness
        X, y = ds[: len(ds)][:2]

        with torch.no_grad():
            z = model.basis(X)

        # surface robustness
        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=p,
            alpha=alpha,
            N=N,
            lim=zlim,
        )

        mu, var = grad.gradient(model.surface, Z, z0=z0, p=p)

        fig, ax = plt.subplots(**fig_kwargs)
        
        ax.quiver(
            Z.cpu()[:, d0],
            Z.cpu()[:, d1],
            mu.cpu()[:, d0],
            mu.cpu()[:, d1],
            fmu,
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=log,
            d0=0,
            d1=1,
            image=False,
            mask=False,
            fig=fig,
            ax=ax,
            contour_kwargs=dict(alpha=0.6),
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs
        )

        if cbar_title is not None:
            fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

        plt.savefig(output[0], bbox_inches="tight")
