
rule effects_pairplot:
    """
    Pairplot of mutational effects
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/effects-pairplot.png"
    run:
        import seaborn as sns

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/effects/{wildcards.ds}-{wildcards.phenotype}/{pth}",
                default=default,
            )

        K = dsget("K", 8)

        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", K
        )
        model.eval()

        plt.figure(figsize=(2, 2), dpi=300)
        ax = plt.subplot(111, polar=True)

        K = fget(
            "zdim",
            default=K,
        )

        W = model.basis.W_mu[:, model.basis.order].detach().numpy()

        df = pd.DataFrame(
            {"z{}".format(i + 1): W[:, i] for i in range(K)}, index=ds.tokenizer.tokens
        )
        sns.pairplot(df, kind="kde", plot_kws=dict(fill=True, levels=8), height=3)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

rule retrain_effects_compare:
    input:
        "experiments/{ds}-{phen}/lantern/full/model.pt",
        "experiments/{ds}-{phen}/lantern/full-r{r}/model.pt",
    group: "figure"
    output:
        "figures/{ds}-{phen}/retrain-effects-r{r}.png"
    run:
        import seaborn as sns
        from scipy import stats

        def corrfunc(x, y, **kws):
            r, p = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate("r = {:.3f}\np = {:.3f}".format(r, p),
                        xy=(.6, .15), xycoords=ax.transAxes)

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/effects/{wildcards.ds}-{wildcards.phen}/{pth}",
                default=default,
            )

        K = dsget("K", 8)

        df, ds, m1 = util.load_run(
            wildcards.ds, wildcards.phen, "lantern", "full", K
        )
        _, _, m2 = util.load_run(
            wildcards.ds, wildcards.phen, "lantern", "full", K, slug=f"-r{wildcards.r}"
        )
        m1.eval()
        m2.eval()

        plt.figure(figsize=(2, 2), dpi=300)
        ax = plt.subplot(111, polar=True)

        K = fget(
            "zdim",
            default=K,
        )

        W = m1.basis.W_mu[:, m1.basis.order].detach().numpy()
        df = pd.DataFrame(
            {"z{}-r0".format(i + 1): W[:, i] for i in range(K)}, index=ds.tokenizer.tokens
        )

        W = m2.basis.W_mu[:, m2.basis.order].detach().numpy()
        df = pd.merge(
            df,
            pd.DataFrame(
                {"z{}-r{}".format(i + 1, wildcards.r): W[:, i] for i in range(K)},
                index=ds.tokenizer.tokens,
            ),
            left_index=True,
            right_index=True,
        )

        g = sns.pairplot(
            df,
            kind="kde",
            plot_kws=dict(fill=True, levels=8),
            height=3,
            x_vars=["z{}-r0".format(i + 1) for i in range(K)],
            y_vars=["z{}-r{}".format(i + 1, wildcards.r) for i in range(K)],
        )
        g.map(corrfunc)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

