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

rule surface_uncrop:
    """
    Uncropped surface plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full{rerun}{kernel}/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface-uncrop{highlight,(-highlight-\w*)?}{rerun,(-r.*)?}{kernel,(-kern-.*)?}.png"
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
        # cbar_kwargs = {"shrink": 0.5, "aspect": 100, "fraction": 0.1}
        cbar_kwargs = {}

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

        # find expanded limits
        X, y = ds[: len(ds)][:2]
        y = y[:, p].numpy()
        with torch.no_grad():
            z = model.basis(X)
            z = z[:, model.basis.order]

        if wildcards.highlight == "":
            lims = [
                z[:, 0].min() - 0.1 * (z[:, 0].max() - z[:, 0].min()),
                z[:, 0].max() + 0.1 * (z[:, 0].max() - z[:, 0].min()),
                z[:, 1].min() - 0.1 * (z[:, 1].max() - z[:, 1].min()),
                z[:, 1].max() + 0.1 * (z[:, 1].max() - z[:, 1].min()),
            ]
        else:
            hgh = wildcards.highlight[wildcards.highlight.rfind("-") + 1 :]

            lims = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/highlight/{hgh}",
            )


        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=p,
            lim=lims,
        )

        # fig, axes = plt.subplots(ncols=2, **fig_kwargs)

        S = 10
        # SIZE = (S, 2*S+1)
        SIZE = (2*S, S+1)
        fig = plt.figure(figsize=(4, 7), dpi=300)
        axes = [
            plt.subplot2grid(
                SIZE,
                (0, 0),
                colspan=S,
                rowspan=S,
            ),
            plt.subplot2grid(
                SIZE,
                # (0, S),
                (S, 0),
                colspan=S,
                rowspan=S,
            ),
        ]

        # scatter
        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            fig=fig,
            ax=axes[0],
            log=log,
            image=image,
            mask=mask,
            cbar_kwargs=cbar_kwargs,
            colorbar=False,
            **plot_kwargs
        )

        axes[0].scatter(
            z[:, 0],
            z[:, 1],
            c=y,
            # alpha=0.4,
            rasterized=True,
            vmin=vrange[0],
            vmax=vrange[1],
            norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
            s=0.3,
        )

        axes[0].set_xlabel("")
        axes[0].set_xticks([])

        # reset limits
        axes[0].set_xlim(*lims[:2])
        axes[0].set_ylim(*lims[2:])

        # interval
        fig, norm, cmap, vrange, interval_im = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            fig=fig,
            ax=axes[1],
            log=log,
            image=image,
            mask=mask,
            showInterval=True,
            cbar_kwargs=cbar_kwargs,
            colorbar=False,
            plotOrigin=False,
            **plot_kwargs
        )
        # axes[1].set_ylabel("")
        # axes[1].set_yticks([])

        # surface colorbar
        if log:
            norm = mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1])
        else:
            norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))

        # cbar = fig.colorbar(sm, ax=axes, **cbar_kwargs)
        cbar = fig.colorbar(
            sm,
            # cax=plt.subplot2grid(
            #     SIZE, (1, 2 * S), colspan=1, rowspan=S // 3,
            # ),
            cax=plt.subplot2grid(
                SIZE, (1, S), colspan=1, rowspan=2 * S // 3,
            ),
            **cbar_kwargs
        )

        if cbar_title is not None:
            fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

        # interval colorbar
        # fig.colorbar(interval_im, ax=fig.axes, **cbar_kwargs)
        fig.colorbar(
            interval_im,
            # cax=plt.subplot2grid((4, 10), (0, 9), colspan=1, rowspan=4,),
            # cax=plt.subplot2grid(
            #     (S, 2 * S + 1), (2 * S // 3 + 1, 2 * S), colspan=1, rowspan=S // 3,
            # ),
            cax=plt.subplot2grid(
                SIZE, (1+S, S), colspan=1, rowspan=2 * S // 3,
            ),
            **cbar_kwargs
        )
        fig.axes[-1].set_title("interval\nwidth", y=1.04, loc="left", ha="left")

        # finish
        plt.savefig(output[0], bbox_inches="tight", verbose=False)

rule surface_slice:
    """
    Surface slice of lantern model.
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

rule surface_scan:
    """
    Surface scan of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full{rerun}{kernel}/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface-scan{rerun,(-r.*)?}{kernel,(-kern-.*)?}.png"
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
        zdim = get(
            config,
            f"figures/effects/{wildcards.ds}-{wildcards.phenotype}/zdim",
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

        # 
        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=p,
            alpha=alpha,
        )

        # number of scans along both dims
        N1 = N2 = 5
        if zdim < 4:
            N2 = 1

        scan1 = 2
        scan2 = 3

        # build the groups for the first scan dimension
        edge1 = np.linspace(z[:, scan1].min(), z[:, scan1].max(), N1 + 2)
        edge1 = edge1[1:-1]
        ind1 = np.digitize(z[:, scan1], edge1)

        # build the groups for the second scan dimension
        edge2 = np.linspace(z[:, scan2].min(), z[:, scan2].max(), N2 + 2)
        edge2 = edge2[1:-1]
        ind2 = np.digitize(z[:, scan2], edge2)

        fixed = [None] * model.basis.K

        # get total surface range
        vrange = (np.inf, -np.inf)
        for i in range(N1):
            for j in range(N2):
                ax = plt.subplot(N2, N1, i + 1 + j * N1)

                fixed[scan1] = edge1[i]
                fixed[scan2] = edge2[j]

                z, fmu, fvar, Z1, Z2, y, Z  = util.buildLandscape(
                    model,
                    ds,
                    mu=df[raw].mean() if raw is not None else 0,
                    std=df[raw].std() if raw is not None else 1,
                    log=log,
                    p=p,
                    fixed=fixed,
                    alpha=alpha,
                )

                vrange = (min(vrange[0], fmu.min()), max(vrange[1], fmu.max()))

        # make plot
        fig = plt.figure(figsize=(3 * N1, 3 * N2),)
        for i in range(N1):
            for j in range(N2):
                ax = plt.subplot(N2, N1, i + 1 + j * N1)

                fixed[scan1] = edge1[i]
                fixed[scan2] = edge2[j]

                z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
                    model,
                    ds,
                    mu=df[raw].mean() if raw is not None else 0,
                    std=df[raw].std() if raw is not None else 1,
                    log=log,
                    p=p,
                    fixed=fixed,
                    alpha=alpha,
                )

                fig, norm, cmap, vrange = util.plotLandscape(
                    z[(ind1 == i) & (ind2 == j), :],
                    fmu,
                    fvar,
                    Z1,
                    Z2,
                    log=log,
                    image=False,
                    mask=False,
                    fig=fig,
                    ax=ax,
                    vrange=vrange,
                )

                plt.scatter(
                    z[(ind1 == i) & (ind2 == j), 0],
                    z[(ind1 == i) & (ind2 == j), 1],
                    c=y[(ind1 == i) & (ind2 == j)],
                    alpha=0.4,
                    rasterized=True,
                    vmin=vrange[0],
                    vmax=vrange[1],
                    norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],)
                    if log
                    else None,
                    s=0.3,
                )

                # reset limits
                plt.xlim(Z1.min(), Z1.max())
                plt.ylim(Z2.min(), Z2.max())

                if N2 > 1:
                    plt.title(f"$z_{scan1+1}={edge1[i]:.3f}$, $z_{scan2+1}={edge2[j]:.3f}$")
                else:
                    plt.title(f"$z_{scan1+1}={edge1[i]:.3f}$")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=True)
