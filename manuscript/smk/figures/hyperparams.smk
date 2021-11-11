
rule lantern_vary_k:
    input:
        lantern = lambda wildcards: (
            expand(
                "experiments/{ds}-{phenotype}/lantern-K{k}/cv{cv}/pred-val.csv",
                cv=range(4),
                k = [1, 2, 3, 4, 5, 6, 8, 16],
                allow_missing=True,
            )
        ),
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/lantern-vary-k-{phen}.png"
    params:
        noiseless = (
            lambda wc: get(
                config,
                f"{wc.ds}/errors/{wc.phenotype}",
                default=["dummy-var"]  # build a default list of errors if none there
                * len(get(config, f"{wc.ds}/phenotypes/{wc.phenotype}")),
            )[
                (
                    get(config, f"{wc.ds}/phenotypes/{wc.phenotype}")
                    .index(wc.phen)
                )
            ]
            == "dummy-var"  # noiseless if no errors
        )
    run:
        scores = None
        reg = re.compile("K(?P<k>\d+)/cv(?P<cv>\d+)")

        # column of targeted phenotype
        ind = (
            get(config, f"{wildcards.ds}/phenotypes/{wildcards.phenotype}")
            .index(wildcards.phen)
        )

        for inp in input.lantern:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(
                inp, i=ind, noiseless=params.noiseless
            ).assign(
                k = int(mtch.group("k")), cv=int(mtch.group("cv")), model="LANTERN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        plot = (
            ggplot(scores, aes(x="k", y="metric",))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("K")
            + theme_matplotlib()
            + theme(figure_size=(2, 2), dpi=300)
        )
        plt.semilogx()

        plot.save(output[0], bbox_inches="tight", verbose=False)
