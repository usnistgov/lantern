rule calibration:
    """
    Calibration plot of model across cross-validated folds.
    """

    input:
        expand("experiments/{dataset}-{phenotype}/{model}/cv{cv}/pred-val.csv", cv=range(10), allow_missing=True)
    output:
        "figures/{dataset}-{phenotype}/{model}/cv-calibration.png"
    group: "figure"
    run:
        scores = pd.concat([pd.read_csv(pth) for pth in input])
        D = len(get(config, f"{wildcards.dataset}/phenotypes/{wildcards.phenotype}"))
        # noise = get(config, f"{wildcards.dataset}/errors/{wildcards.phenotype}", default=None)

        plt.figure(figsize=(3 * D, 2))
        for d in range(D):
            plt.subplot(1, D, d + 1)

            if wildcards.model == "globalep":
                y = scores.observed_phenotype
                yh = scores.func_score
            else:
                yh = scores[f"yhat{d}"]
                y = scores[f"y{d}"]

            plt.hist2d(yh, y, bins=30, norm=mpl.colors.LogNorm())

            rng = [min(y.min(), yh.min()), max(y.max(), yh.max())]
            plt.plot(rng, rng, c="r")

        plt.savefig(output[0], bbox_inches="tight", verbose=False)
