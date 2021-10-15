
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
        reg = re.compile("/(?P<label>.*)/history.csv")
        for h in input:

            mtch = reg.search(h)
            tmp = pd.read_csv(h).assign(transform=mtch.group("label"))

            hist = pd.concat((hist, tmp), axis=0)

        hist = pd.melt(
            hist,
            ["epoch", "transform"],
            ["variational_basis-train", "neg-loglikelihood-train", "gp-kl-train",],
            var_name="metric",
        )

        plot = (
            ggplot(hist, aes(x="epoch", y="value", color="transform",),)
            + facet_wrap("metric", nrow=1, scales="free_y")
            + geom_line()
            + theme_matplotlib()
            + ylab("")
            + theme(figure_size=(6, 2), dpi=300, subplots_adjust={"wspace": 0.5})
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
