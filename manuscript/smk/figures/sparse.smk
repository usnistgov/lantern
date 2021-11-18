
rule sparse_effects:
    """
    Sparsity of mutational effects
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/effects-sparsity.png"
    run:
        import seaborn as sns
        from scipy.stats import chi2

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

        K = fget("zdim", default=K,)

        Wmu = model.basis.W_mu[:, model.basis.order[:K]].detach()
        Wvar = model.basis.W_log_sigma[:, model.basis.order[:K]].detach().exp().pow(2)
        tstat = (Wmu.pow(2) / Wvar).sum(axis=1)

        alphas = np.logspace(-4, -1)

        # percentage removed at this significance level
        percents = [
            (chi2(K).cdf(tstat) < 1 - alpha).sum() / ds.p for alpha in alphas
        ]

        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(alphas, percents)
        plt.ylabel("effective zero mutations")
        plt.xlabel("confidence level")
        plt.semilogx()
        plt.grid()

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
