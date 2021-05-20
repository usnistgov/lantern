rule anglehist:
    """
    Angle histogram of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/anglehist.png"

    run:

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", dsget("K", 8)
        )
        model.eval()

        plt.figure(figsize=(2, 2), dpi=300)
        ax = plt.subplot(111, polar=True)

        W = model.basis.W_mu[:, model.basis.order].detach().numpy()

        # "y" is the first argument, "x" is second
        theta = np.arctan2(W[:, 1], W[:, 0])

        H, edges = np.histogram(theta, bins=100, density=True)

        ax = plt.subplot(111, polar="true")

        bottom = H.max() * 0.5

        bars = ax.bar(
            edges[1:],
            H,
            width=edges[1:] - edges[:-1],
            bottom=bottom,
            zorder=100,
        )
        ax.set_yticklabels([])

        for h, b in zip(H, bars):
            b.set_edgecolor("k")
            b.set_facecolor(plt.cm.viridis(h / H.max()))

        plt.savefig(output[0], bbox_inches="tight")
