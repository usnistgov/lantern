rule diffops:
    input:
        "experiments/laci-joint/lantern/full/model.pt"
    output:
        "figures/diffops-curvature.png",
        "figures/diffops-slope.png"
    group: "figure"
    run:
        df, ds, model = util.load_run("laci", "joint", "lantern", "full", 8)

        model.eval()
        X = ds[: len(ds)][0]

        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        from lantern.diffops import lapl
        from lantern.diffops import grad
        from lantern.diffops import metric

        zrng = torch.linspace(-4.8, 4.8)
        Zpred = torch.zeros(100, 8)
        Zpred[:, model.basis.order[0]] = zrng

        strat = model.surface.variational_strategy
        if isinstance(strat, IndependentMultitaskVariationalStrategy):
            strat = strat.base_variational_strategy
        z0 = strat.inducing_points.detach()
        if z0.ndim > 2:
            z0 = z0[0, :, :]

        with torch.no_grad():

            fpred = model.surface(Zpred)
            lower, upper = fpred.confidence_region()

            lmu, lvar = lapl.laplacian(
                model.surface, Zpred, z0, dims=model.basis.order[:2], p=0
            )
            nrm = torch.distributions.Normal(lmu, lvar.sqrt())
            additivity = metric.kernel(lmu, lvar)

            lmu = lmu.numpy()
            lvar = lvar.numpy()

        fig, ax = plt.subplots(dpi=200, figsize=(3, 2))
        plt.plot(zrng, fpred.mean[:, 0], label="$f(z)$")
        plt.fill_between(zrng, lower[:, 0], upper[:, 0], alpha=0.6)
        plt.xlabel("$z_1$")

        plt.setp([ax.get_yticklabels()], color="C0")
        ax.tick_params(axis="y", color="C0")
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="C0", linewidth=1.0)
        ax.set_ylabel("$f(\mathbf{z})$", color="C0")

        ax = plt.twinx()
        plt.plot(zrng, lmu, c="C1", label="Laplacian")
        plt.fill_between(
            zrng,
            lmu - np.sqrt(lvar) * 2,
            lmu + np.sqrt(lvar) * 2,
            alpha=0.6,
            color="C1",
        )
        plt.axhline(0, c="C1", ls="--")

        plt.setp([ax.get_yticklabels()], color="C1")
        ax.tick_params(axis="y", color="C1")
        for pos in [
            "right",
        ]:
            plt.setp(ax.spines[pos], color="C1", linewidth=1.0)
        ax.set_ylabel("curvature", color="C1", rotation=270)

        plt.savefig(output[0], bbox_inches="tight", verbose=False)

        """Slope
        """

        d0 = model.basis.order[0]
        with torch.no_grad():

            fpred = model.surface(Zpred)
            lower, upper = fpred.confidence_region()

            lmu, lvar = grad.gradient(model.surface, Zpred, z0, p=0)
            lmu, lvar = lmu[:, d0, 0], lvar[:, d0, d0]
            nrm = torch.distributions.Normal(lmu, lvar.sqrt())

            lmu = lmu.numpy()
            lvar = lvar.numpy()

        fig, ax = plt.subplots(dpi=200, figsize=(3, 2))
        plt.plot(zrng, fpred.mean[:,0], label="$f(z)$")
        plt.fill_between(zrng, lower[:,0], upper[:,0], alpha=0.6)
        plt.xlabel("$z_1$")

        plt.setp([ax.get_yticklabels()], color="C0")
        ax.tick_params(axis="y", color="C0")
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="C0", linewidth=1.0)
        ax.set_ylabel("$f(\mathbf{z}$)", color="C0")

        ax = plt.twinx()
        plt.plot(zrng, lmu, c="C2", label="Laplacian")
        plt.fill_between(zrng, lmu - np.sqrt(lvar)*2, lmu + np.sqrt(lvar)*2, alpha=0.6, color="C2")
        plt.axhline(0, c="C2", ls="--")

        plt.setp([ax.get_yticklabels()], color="C2")
        ax.tick_params(axis="y", color="C2")
        for pos in [
            "right",
        ]:
            plt.setp(ax.spines[pos], color="C2", linewidth=1.0)
        ax.set_ylabel("slope", color="C2", rotation=270)

        plt.savefig(output[1], bbox_inches="tight", verbose=False)

rule diffops_distance:
    """
    Distance plot of lantern diffops metric.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/{target}/diffops-distance.png"

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
        import matplotlib.gridspec as gridspec

        from lantern.diffops import lapl, grad
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


        mu, var = grad.gradient(model.surface, Z, z0=z0, p=p)
        robustness_surf = metric.kernel(mu, var)

        mu, var = lapl.laplacian(
            model.surface, Z, z0=z0, dims=model.basis.order[:dims], p=p
        )
        additivity_surf = metric.kernel(mu, var)

        W = model.basis.W_mu.detach().numpy()

        fig = plt.figure(figsize=(4, 6), dpi=200)

        w = 3
        h = 2
        s = 3
        gs = gridspec.GridSpec(s * (2 * h + 1), s * w + 1, hspace=1.4*s)

        i = torch.argmax(robustness_surf)

        dist = np.zeros(additivity_surf.shape[0])
        for j in range(additivity_surf.shape[0]):
            # dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)
            dist[j] = (Z[i, :] - Z[j, :]).norm().item()

        ax_rob = plt.subplot(gs[0 : s * h, : s * w])
        _, _, _, im_rob = plt.hist2d(
            dist,
            robustness_surf.numpy(),
            bins=(np.linspace(0, 5, 30), np.linspace(0, 1, 30)),
            norm=mpl.colors.LogNorm(),
        )

        plt.ylabel("Robustness")

        i = torch.argmax(additivity_surf)

        dist = np.zeros(additivity_surf.shape[0])
        for j in range(additivity_surf.shape[0]):
            # dist[j] = mahalanobis(Z[i, :].numpy(), Z[j, :].numpy(), Sigma_inv)
            dist[j] = (Z[i, :] - Z[j, :]).norm().item()

        ax_add = plt.subplot(gs[s * h : 2 * s * h, : s * w])
        _, _, _, im_add = plt.hist2d(
            dist,
            additivity_surf.numpy(),
            bins=(np.linspace(0, 5, 30), np.linspace(0, 1, 30)),
            norm=mpl.colors.LogNorm(),
        )

        plt.xlabel("Distance")
        plt.ylabel("Additivity")

        ax_hist = plt.subplot(gs[-s:, : s * w])
        ax_hist.hist(
            [torch.from_numpy(W[i, :]).norm().item() for i in range(W.shape[0])],
            bins=np.linspace(0, 5, 50),
            color="k",
        )
        plt.xlim(0, 5)
        plt.ylabel("count")
        plt.xlabel("magnitude")

        ax_rob_cb = plt.subplot(gs[0 : s * h, s * w])
        plt.colorbar(im_rob, cax=ax_rob_cb)

        ax_add_cb = plt.subplot(gs[s * h : 2 * s * h, s * w])
        plt.colorbar(im_add, cax=ax_add_cb)

        # ax_rob.set_xticks([])

        plt.savefig(output[0], bbox_inches="tight")
