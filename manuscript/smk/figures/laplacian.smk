from gpytorch.variational import IndependentMultitaskVariationalStrategy

rule laplacian:
    """
    Laplacian plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/laplacian.png"

    run:
        from lantern.diffops import lapl
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
        dims = fget(
            "dims",
            default=8,
        )
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

        mu, var = lapl.laplacian(
            model.surface, Z, z0=z0, dims=model.basis.order[:dims], p=p
        )


        image = mu
        image = image.reshape(Z1.shape)

        fig, ax = plt.subplots(**fig_kwargs)
        vmin = -abs(mu).max()
        vmax = abs(mu).max()

        cmap = "PuOr"

        im = ax.imshow(
            image,
            alpha=0.9,
            extent=(Z[:, d0].min(), Z[:, d0].max(), Z[:, d1].min(), Z[:, d1].max()),
            origin="lower"
            if zlim is not None
            else "upper",  # this is a guess on the right way to do it, not sure why it is though
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="lanczos",
        )

        fig.colorbar(im, ax=ax, **cbar_kwargs)
        fig.axes[-1].set_title("Laplacian", y=1.04, loc="left", ha="left")
        
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
