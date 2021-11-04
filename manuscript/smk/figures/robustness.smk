from gpytorch.variational import IndependentMultitaskVariationalStrategy

rule robustness:
    """
    Gradient plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/robustness.png"

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

        mu, var = grad.gradient(model.surface, Z, z0=z0, p=p)
        robustness_surf = metric.kernel(mu, var)

        fig, ax = plt.subplots(**fig_kwargs)
        
        vmin = 0
        vmax = 1

        im = ax.imshow(
            robustness_surf.reshape(Z1.shape).numpy(),
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
                vmin=robustness_surf.min(),
                vcenter=(robustness_surf.max() - robustness_surf.min()) * midpoint
                + robustness_surf.min(),
                vmax=robustness_surf.max(),
            )
            if midpoint is not None
            else None,
        )

        fig.colorbar(im, ax=ax, **cbar_kwargs)
        if cbar_title is not None:
            fig.axes[-1].set_title(
                "robustness", y=1.04, loc="left", ha="left", style="italic",
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

        # plt.tight_layout()

        plt.savefig(output[0], bbox_inches="tight")

rule robustness_diagnostic:
    """
    Diagnostic plot of lantern robustness metric.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/robustness-diagnostic.png"

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

        mu, var = grad.gradient(model.surface, Zds, z0=z0, p=p)
        robustness_ds = metric.kernel(mu, var)

        # # surface robustness
        # z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
        #     model,
        #     ds,
        #     mu=df[raw].mean() if raw is not None else 0,
        #     std=df[raw].std() if raw is not None else 1,
        #     log=log,
        #     p=p,
        #     alpha=alpha,
        #     N=N,
        #     lim=zlim
        # )

        # mu, var = grad.gradient(model.surface, Z, z0=z0, p=p)
        # robustness_surf = metric.kernel(mu, var)

        # Mutational effects
        W = model.basis.W_mu.detach()
        
        # number of bins
        B = 10
        quants = torch.linspace(0, 1, B)
        dist = []
        vals = []
        for q in quants:

            # find 10 closest index to this quantile
            ind = torch.argsort((q - robustness_ds).abs())[:10]

            tmp = []
            for i in ind:

                # centered point
                Zq = Zds[[i], :].repeat(W.shape[0] + 1, 1)

                # add mutations to non-starting values
                Zq[1:, :] += W

                with torch.no_grad():
                    fq = model.surface(Zq)

                # average change
                tmp.append((fq.mean[1:, p] - fq.mean[0, p]).mean().item())

            vals.append(q.item())
            dist.append(tmp)
            # dist.append((fq.mean[1:, p] - fq.mean[0, p]).numpy())

        plt.figure(figsize=(3, 2), dpi=300)

        # robustness vs change in surface (+- standard error)
        plt.errorbar(
            vals,
            np.array(dist).mean(axis=1),
            # np.array(dist).std(axis=1) / (len(dist[0]) ** 0.5),
            np.array(dist).std(axis=1),
        )

        # plt.boxplot(dist, positions=vals)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

rule robustness_distance:
    """
    Distance plot of lantern robustness metric.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/robustness-distance.png"

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

        mu, var = grad.gradient(model.surface, Z, z0=z0, p=p)
        robustness_surf = metric.kernel(mu, var)

        # for mdist
        Sigma_inv = (model.basis.qalpha(detach=True).mean).diag().numpy()
        W = model.basis.W_mu.detach().numpy()

        i = torch.argmax(robustness_surf)

        # mn = []
        # for i in range(robustness_surf.shape[0]):
        #     # compute mahalanobis distance
        #     dist = np.zeros(robustness_surf.shape[0])
        #     for j in range(robustness_surf.shape[0]):
        #         dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)

        #     # points within expected mahalanobis distance
        #     sel = dist ** 2 < 2

        #     # minimum metric value
        #     mn.append(robustness_surf[sel].min())

        plt.figure(figsize=(3, 2), dpi=300)

        # plt.hist2d(robustness_surf.numpy(), mn, bins=30, norm=mpl.colors.LogNorm())

        dist = np.zeros(robustness_surf.shape[0])
        for j in range(robustness_surf.shape[0]):
            dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)

        plt.hist2d(
            dist ** 2,
            robustness_surf.numpy(),
            bins=(np.linspace(0, 5, 30), np.linspace(0, 1, 30)),
            norm=mpl.colors.LogNorm(),
        )
        plt.axvline(
            np.mean(
                [
                    mahalanobis(np.zeros(8), W[i, :], Sigma_inv)
                    for i in range(W.shape[0])
                ]
            ),
            c="r",
            alpha=0.2,
        )

        plt.colorbar()

        plt.xlabel("Mahalanobis distance")
        plt.ylabel("robustness")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
