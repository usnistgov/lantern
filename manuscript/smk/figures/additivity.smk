from gpytorch.variational import IndependentMultitaskVariationalStrategy

rule additivity:
    """
    Laplacian plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/additivity.png"

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
                f"figures/laplacian/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/{pth}",
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
        
        # figure parameters
        dims = fget(
            "dims",
            default=8,
        )
        N = fget(
            "N",
            default=50,
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
        )

        mu, var = lapl.laplacian(
            model.surface, Z, z0=z0, dims=model.basis.order[:dims], p=p
        )
        additivity_surf = metric.kernel(mu, var)


        image = mu
        image = image.reshape(Z1.shape)

        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        midpoint = None
        vmin = 0
        vmax = 1

        cmap = "PuOr"

        im = ax.imshow(
            additivity_surf.reshape(Z1.shape).numpy(),
            extent=(Z1.min(), Z1.max(), Z2.min(), Z2.max()),
            aspect="auto",
            cmap="Greys_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="lanczos",
            origin="upper",
            norm=MidpointNormalize(
                vmin=robustness_surf.min(),
                vcenter=(robustness_surf.max() - robustness_surf.min()) * midpoint
                + robustness_surf.min(),
                vmax=robustness_surf.max(),
            )
            if midpoint is not None
            else None,
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
        )

        fig.colorbar(im, ax=ax)

        plt.savefig(output[0], bbox_inches="tight")
