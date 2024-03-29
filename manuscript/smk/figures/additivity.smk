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

        class MidpointNormalize(mpl.colors.Normalize):
            def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
                self.vcenter = vcenter
                mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

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
        midpoint = fget(
            "midpoint",
            default=None,
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
        additivity_surf = metric.kernel(mu, var)


        fig, ax = plt.subplots(**fig_kwargs)
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
            origin="lower"
            if zlim is not None
            else "upper",  # this is a guess on the right way to do it, not sure why it is though
            norm=MidpointNormalize(
                vmin=additivity_surf.min(),
                vcenter=(additivity_surf.max() - additivity_surf.min()) * midpoint
                + additivity_surf.min(),
                vmax=additivity_surf.max(),
            )
            if midpoint is not None
            else None,
        )

        fig.colorbar(im, ax=ax, **cbar_kwargs)
        fig.axes[-1].set_title(
            "additivity", y=1.04, loc="left", ha="left", style="italic",
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

rule additivity_distance:
    """
    Distance plot of lantern additivity metric.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/additivity-distance.png"

    run:
        class MidpointNormalize(mpl.colors.Normalize):
            def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
                self.vcenter = vcenter
                mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        from scipy.spatial.distance import mahalanobis

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
        midpoint = fget(
            "midpoint",
            default=None,
        )

        cbar_kwargs.update(fget("cbar_kwargs", default={}))
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

            Zds = model.basis(X)

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
            lim=zlim
        )

        mu, var = lapl.laplacian(
            model.surface, Z, z0=z0, dims=model.basis.order[:dims], p=p
        )
        additivity_surf = metric.kernel(mu, var)

        # for mdist
        Sigma_inv = (model.basis.qalpha(detach=True).mean).diag().numpy()
        W = model.basis.W_mu.detach().numpy()

        i = torch.argmax(additivity_surf)

        # mn = []
        # for i in range(additivity_surf.shape[0]):
        #     # compute mahalanobis distance
        #     dist = np.zeros(additivity_surf.shape[0])
        #     for j in range(additivity_surf.shape[0]):
        #         dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)

        #     # points within expected mahalanobis distance
        #     sel = dist ** 2 < 2

        #     # minimum metric value
        #     mn.append(additivity_surf[sel].min())

        plt.figure(figsize=(3, 2), dpi=300)

        # plt.hist2d(additivity_surf.numpy(), mn, bins=30, norm=mpl.colors.LogNorm())

        dist = np.zeros(additivity_surf.shape[0])
        for j in range(additivity_surf.shape[0]):
            # dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)
            dist[j] = (Z[i, :] - Z[j, :]).norm().item()

        plt.hist2d(
            # dist ** 2,
            dist,
            additivity_surf.numpy(),
            bins=(np.linspace(0, 5, 30), np.linspace(0, 1, 30)),
            norm=mpl.colors.LogNorm(),
        )

        # plt.axvline(
        #     # np.mean(
        #     #     [
        #     #         mahalanobis(np.zeros(8), W[i, :], Sigma_inv)
        #     #         for i in range(W.shape[0])
        #     #     ]
        #     # ),
        #     np.mean(
        #         [torch.from_numpy(W[i, :]).norm().item() for i in range(W.shape[0])]
        #     ),
        #     c="r",
        #     alpha=0.4,
        # )

        plt.colorbar()

        plt.xlabel("Distance")
        plt.ylabel("Additivity")

        plt.twinx()
        plt.hist(
            [torch.from_numpy(W[i, :]).norm().item() for i in range(W.shape[0])],
            bins=30,
            color="k",
            alpha=0.6
        )

        # plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
