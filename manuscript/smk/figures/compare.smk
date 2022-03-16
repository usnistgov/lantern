
rule globalep_compare:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "experiments/{ds}-{phenotype}/globalep/cv0/model.pkl",
    output:
        "figures/{ds}-{phenotype}/globalep-compare.png"
    run:
        from scipy.stats import pearsonr

        # alpha = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/alpha",
        #     default=0.01,
        # )
        raw = get(
            config,
            f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.phenotype}/raw",
            default=None,
        )
        # log = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/log",
        #     default=False,
        # )
        # p = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/p",
        #     default=0,
        # )
        # image = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/image",
        #     default=False,
        # )
        # scatter = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/scatter",
        #     default=True,
        # )
        # mask = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/mask",
        #     default=False,
        # )
        # cbar_kwargs = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_kwargs",
        #     default={},
        # )
        # fig_kwargs = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/fig_kwargs",
        #     default=dict(dpi=300, figsize=(4, 3)),
        # )
        # cbar_title = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/cbar_title",
        #     default=None,
        # )
        # plot_kwargs = get(
        #     config,
        #     f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/plot_kwargs",
        #     default={},
        # )
        # zdim = get(
        #     config,
        #     f"figures/effects/{wildcards.ds}-{wildcards.phenotype}/zdim",
        # )

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        correlation = True
        p = 0

        df, ds, model = util.load_run(
            wildcards.ds,
            wildcards.phenotype,
            "lantern",
            "full",
            dsget("K", 8),
        )
        model.eval()

        glep = pickle.load(open(input[3], "rb"))

        # compare latent effects
        W = pd.DataFrame(
            {
                "mutation": ds.tokenizer.tokens,
                "z0": model.basis.W_mu.detach().numpy()[:, model.basis.order[0]],
            }
        )

        # fix leading "S" for merge
        if wildcards.ds == "gfp":
            W["mutation"] = W.mutation.str[1:]

        mrg = pd.merge(W, glep.latent_effects_df, on="mutation")

        # map from glep to LANTERN
        beta, _, _, _ = np.linalg.lstsq(
            mrg.z0.values[:, None], mrg.latent_effect, rcond=None
        )
        beta = beta[0]

        # compare functions
        zspline = np.linspace(
            glep.phenotypes_df.latent_phenotype.min(),
            glep.phenotypes_df.latent_phenotype.max(),
            100,
        )

        X = ds[: len(ds)][0]
        d0 = model.basis.order[0]
        with torch.no_grad():
            z = model.basis(X)

            Z = torch.zeros(100, 8)
            Z[:, d0] = torch.linspace(z[:, d0].min(), z[:, d0].max())
            f = model.surface(Z)
            lower, upper = f.confidence_region()

            fmu = f.mean
            if fmu.ndim > 1:
                fmu = fmu[:, p]
                lower = lower[:, p]
                upper = upper[:, p]

        mu = df[raw].mean() if raw is not None else 0
        std = df[raw].std() if raw is not None else 1

        # make plots
        if correlation:
            fig = plt.figure(figsize=(5, 2), dpi=300)
            ax = plt.subplot(121)
        else:
            fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

        plt.hist(beta * z[:, d0].numpy(), bins=30, log=True, color="tab:purple")
        # plt.hist(mrg.latent_effect, bins=30, log=True, color="tab:pink")
        plt.ylabel("variant count", color="tab:purple")
        plt.setp([ax.get_yticklabels()], color="tab:purple")
        ax.tick_params(axis="y", color="tab:purple")
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="tab:purple", linewidth=1.0)

        plt.xlabel("$z_1$ (I-spline)")

        ax = plt.twinx()
        plt.plot(
            zspline, mu + std * glep.epistasis_func(zspline), c="k", label="I-spline"
        )
        plt.plot(beta * Z[:, d0], mu + std * fmu, c="C0", label="LANTERN")
        plt.fill_between(
            beta * Z[:, d0], mu + std * lower, mu + std * upper, alpha=0.8, color="C0"
        )
        plt.ylabel(
            dsget("phenotype_labels", default=["phenotype"] * (p + 1))[p],
            rotation=-90,
            labelpad=10
        )
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="tab:purple", linewidth=1.0)

        _z = beta * Z[:, d0]
        rng = _z.max() - _z.min()
        plt.xlim(_z.min() - 0.1 * rng, _z.max() + 0.1 * rng)

        # if legend is not None:
        #     plt.legend(loc=legend, fontsize=6)

        # inset
        if correlation:
            iax = plt.subplot(122)
            # iax = ax.inset_axes(pos)
            z0 = np.sign(beta) * mrg.z0
            _, _, _, im = iax.hist2d(
                # z0, mrg.latent_effect, bins=30, norm=mpl.colors.LogNorm()
                z0, mrg.latent_effect, bins=30, cmap="Blues", cmin=1, vmin=0
            )
            # iax.plot(mrg.z0, beta * mrg.z0, c="r")
            iax.plot(z0, np.abs(beta) * z0, c="r", alpha=0.8)
            iax.set_xlabel("$z_1$ LANTERN")
            iax.set_ylabel("$z_1$ I-spline")

            rho, pval = pearsonr(z0, mrg.latent_effect)
            iax.text(
                0.05,
                0.9,
                # r"$\rho = {rho:.3f}$\n$p<{pval}$".format(rho=rho, pval=pval),
                r"$\rho = {rho:.3f}$".format(rho=rho),
                transform=iax.transAxes,
                fontsize=8,
                color="black",
            )
            fig.colorbar(im, ax=iax)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
