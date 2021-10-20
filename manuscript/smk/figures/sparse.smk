
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
        from torch.distributions import Normal

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

        K = fget(
            "zdim",
            default=K,
        )

        qW = Normal(
            model.basis.W_mu[:, model.basis.order[:K]].detach(),
            model.basis.W_log_sigma[:, model.basis.order[:K]].detach().exp(),
        )

        c = torch.min(qW.cdf(torch.tensor(0)), 1-qW.cdf(torch.tensor(0.)))

        alphas = torch.logspace(-4, -1)

        # percentage removed at this significance level
        percents = [
            1 - ((c.prod(dim=1) < alpha / 2).sum() / qW.mean.shape[0]).item()
            for alpha in alphas
        ]

        plt.figure(figsize=(3,2), dpi=300)
        plt.plot(alphas.numpy(), percents)
        plt.ylabel("non-significant mutations")
        plt.xlabel("significance level")
        plt.semilogx()
        plt.grid()

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
