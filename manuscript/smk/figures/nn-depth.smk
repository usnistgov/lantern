rule nn_depth:
    input:
        lantern = lambda wildcards: (
            expand(
                "experiments/{ds}-{phenotype}/lantern/cv{cv}/pred-val.csv",
                cv=range(10),
                allow_missing=True,
            )
        ),
        feedforward = lambda wildcards: (
            expand(
                "experiments/{ds}-{phenotype}/feedforward-K8-D{d}-W32/cv{cv}/pred-val.csv",
                cv=range(10),
                d = range(1, 4),
                allow_missing=True,
            )
        ),
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/nn-depth-{phen}.png"
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
        reg = re.compile("cv(?P<cv>\d+)")

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
                width = 0, cv=int(mtch.group("cv")), model="LANTERN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        reg = re.compile("K8-D(?P<d>\d+)-W32/cv(?P<cv>\d+)")
        for inp in input.feedforward:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(
                inp, i=ind, noiseless=params.noiseless
            ).assign(
                width = int(mtch.group("d")),
                cv=int(mtch.group("cv")),
                model="Feed-forward NN",
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        plot = (
            ggplot(scores, aes(x="width", y="metric", color="model"))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("NN depth")
            + theme_matplotlib()
            + theme(figure_size=(2, 2), dpi=300)
            + scale_x_continuous(breaks=[1, 2, 3], minor_breaks=[])
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
