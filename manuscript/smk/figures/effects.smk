
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
