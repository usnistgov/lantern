rule surface:
    """
    Surface plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full{rerun}{kernel}/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface{rerun,(-r.*)?}{kernel,(-kern-.*)?}.png"
    group: "figure"
    run:

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        alpha = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/alpha",
            default=0.01,
        )
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
        image = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/image",
            default=False,
        )
        scatter = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/scatter",
            default=True,
        )
        mask = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/mask",
            default=False,
        )
        cbar_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_kwargs",
            default={},
        )
        fig_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/fig_kwargs",
            default=dict(dpi=300, figsize=(4, 3)),
        )
        cbar_title = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_title",
            default=None,
        )
        plot_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/plot_kwargs",
            default={},
        )

        df, ds, model = util.load_run(
            wildcards.ds,
            wildcards.phenotype,
            "lantern",
            "full",
            dsget("K", 8),
            slug=wildcards.rerun+wildcards.kernel,
            kernel=wildcards.kernel
        )
        model.eval()

        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=p,
            alpha=alpha,
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=log,
            image=image,
            mask=mask,
            cbar_kwargs=cbar_kwargs,
            fig_kwargs=fig_kwargs,
            **plot_kwargs
        )

        if scatter:
            
            plt.scatter(
                z[:, 0],
                z[:, 1],
                c=y,
                alpha=0.4,
                rasterized=True,
                vmin=vrange[0],
                vmax=vrange[1],
                norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
                s=0.3,
            )

            # reset limits
            plt.xlim(
                np.quantile(z[:, 0], alpha / 2),
                np.quantile(z[:, 0], 1 - alpha / 2),
            )
            plt.ylim(
                np.quantile(z[:, 1], alpha / 2),
                np.quantile(z[:, 1], 1 - alpha / 2),
            )

        if cbar_title is not None:
            fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

        plt.savefig(output[0], bbox_inches="tight", verbose=False)

rule surface_slice:
    """
    Surface plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full{rerun}{kernel}/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface-slice-z{k}{rerun,(-r.*)?}{kernel,(-kern-.*)?}.png"
    group: "figure"
    run:

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        alpha = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/slice-alpha",
            default=0.2,
        )
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
        image = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/image",
            default=False,
        )
        scatter = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/scatter",
            default=True,
        )
        mask = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/mask",
            default=False,
        )
        cbar_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_kwargs",
            default={},
        )
        fig_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/fig_kwargs",
            default=dict(dpi=300, figsize=(4, 3)),
        )
        cbar_title = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_title",
            default=None,
        )
        plot_kwargs = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/plot_kwargs",
            default={},
        )

        df, ds, model = util.load_run(
            wildcards.ds,
            wildcards.phenotype,
            "lantern",
            "full",
            dsget("K", 8),
            slug=wildcards.rerun+wildcards.kernel,
            kernel=wildcards.kernel
        )
        model.eval()

        K = int(wildcards.k) - 1

        X = ds[:len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        Zpred = torch.zeros(100, 8)

        Zpred[:, model.basis.order[K]] = torch.linspace(
            torch.quantile(Z[:, K], alpha/2), torch.quantile(Z[:, K], 1-alpha/2)
        )
        print(
            alpha,
            torch.quantile(Z[:, K], alpha / 2),
            torch.quantile(Z[:, K], 1 - alpha / 2),
            Z[:, K].min(),
            Z[:, K].max(),
        )

        for z1 in torch.linspace(torch.quantile(Z[:, 0], 0.1), torch.quantile(Z[:, 0], 0.9), 10):
            Zpred[:, model.basis.order[0]] = z1

            with torch.no_grad():
                f = model.surface(Zpred)
                lo, hi = f.confidence_region()

            plt.plot(
                Zpred[:, model.basis.order[K]].numpy(),
                f.mean[:, p].numpy() if f.mean.ndim > 1 else f.mean.numpy(),
                label=f"$z_1$ = {z1:0.3f}",
            )
            plt.fill_between(
                Zpred[:, model.basis.order[K]].numpy(),
                lo[:, p].numpy() if lo.ndim> 1 else lo.numpy(),
                hi[:, p].numpy() if hi.ndim> 1 else hi.numpy(),
                alpha=0.4,
            )

        plt.xlabel(f"$z_{K+1}$")
        plt.ylabel(config[wildcards.ds]["phenotype_labels"][p])
        plt.legend()

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=False)

rule posterior_sample_surface:
    """Sample from learned posterior parameters"""
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface-sample.png"
    group: "figure"
    run:

        from gpytorch.distributions import MultivariateNormal

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        p = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/p",
            default=0,
        )

        df, ds, model = util.load_run(
            wildcards.ds,
            wildcards.phenotype,
            "lantern",
            "full",
            dsget("K", 8),
        )
        model.eval()

        X = ds[:len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)[:, model.basis.order]

        Z1 = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 50)
        Z2 = np.linspace(Z[:, 1].min(), Z[:, 1].max(), 50)
        ZZ1, ZZ2 = np.meshgrid(Z1, Z2)

        Zsamp = torch.zeros(2500, 8)
        Zsamp[:, 0] = torch.from_numpy(ZZ1.ravel())
        Zsamp[:, 1] = torch.from_numpy(ZZ2.ravel())

        Kz = model.surface.kernel(Zsamp).evaluate()
        if ds.D > 1:
            Kz = Kz[p, :, :]

        
        plt.figure(figsize=(3, 6), dpi=200)

        for i in range(3):
            plt.subplot(3, 1, i+1)
            f = MultivariateNormal(torch.zeros(2500), Kz+torch.eye(2500)*1e-4).sample()
            plt.contourf(ZZ1, ZZ2, f.reshape(ZZ1.shape).numpy(), levels=8,)

            if i == 2:
                plt.xlabel("$z_1$")
            plt.ylabel("$z_2$")

            plt.colorbar()

        plt.tight_layout()
        plt.savefig(output[0])

