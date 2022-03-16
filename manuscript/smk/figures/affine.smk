
rule affine_history:
    """
    History of training after affine transformations
    """

    input:
        lambda wc: expand(
            "experiments/{ds}-{phen}/lantern/affine/{label}/history.csv",
            label=get(config, f"{wc.ds}/lantern/affine").keys(),
            allow_missing = True
        )
    group: "figure"
    output:
        "figures/{ds}-{phen}/affine-history.png"
    run:
        
        hist = pd.DataFrame()
        for h in input:

            label = h.split("/")[-2]
            tmp = pd.read_csv(h).assign(transform=label)

            hist = pd.concat((hist, tmp), axis=0)

        hist = pd.melt(
            hist,
            ["epoch", "transform"],
            ["variational_basis-train", "neg-loglikelihood-train", "gp-kl-train",],
            var_name="metric",
        )

        breaks = [hist.epoch.min(), (hist.epoch.max() - hist.epoch.min())/2, hist.epoch.max() + 1]
        plot = (
            ggplot(hist, aes(x="epoch", y="value", color="transform",),)
            + facet_wrap("metric", nrow=1, scales="free_y")
            + geom_line()
            + theme_matplotlib()
            + ylab("")
            + theme(figure_size=(6, 2), dpi=300, subplots_adjust={"wspace": 0.5})
            + scale_x_continuous(breaks = breaks)
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)

rule affine_effects_compare:
    input:
        lambda wc: expand(
            "experiments/{ds}-{phen}/lantern/affine/{label}/model.pt",
            label=get(config, f"{wc.ds}/lantern/affine").keys(),
            allow_missing = True
        )
    group: "figure"
    output:
        "figures/{ds}-{phen}/affine-effects.png"
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

        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phen, "lantern", "full", K
        )
        model.eval()

        Kmax = 2 # just first two dims
        # fget(
        #     "zdim",
        #     default=K,
        # )

        W = model.basis.W_mu[:, model.basis.order].detach().numpy()

        # merge this against each
        template = pd.DataFrame(
            {"z{}".format(i + 1): W[:, i] for i in range(Kmax)},
        ).assign(token=ds.tokenizer.tokens)
        template = pd.melt(
            template,
            id_vars = ["token"],
            value_vars=[f"z{z+1}" for z in range(Kmax)],
            var_name="dimension",
            value_name="original",
        )

        df = pd.DataFrame()
        for pt in input:
            spl = pt.split("/")
            label = spl[-2]

            model.load_state_dict(torch.load(pt, "cpu"))

            W = model.basis.W_mu[:, model.basis.order].detach().numpy()

            # merge this against each
            tmp = pd.DataFrame(
                {"z{}".format(i + 1): W[:, i] for i in range(Kmax)},
            ).assign(token=ds.tokenizer.tokens)
            tmp = pd.melt(
                tmp,
                id_vars = ["token"],
                value_vars=[f"z{z+1}" for z in range(Kmax)],
                var_name="dimension",
                value_name="new",
            ).assign(transform=label)

            mrg = pd.merge(template, tmp, on=["dimension", "token"])
            df = pd.concat((df, mrg), axis=0)

        g = sns.FacetGrid(df, row="dimension", col="transform", sharex=False, sharey=False)
        g.map(sns.kdeplot, "original", "new", fill=True, levels=8)
        g.map(corrfunc, "original", "new")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
