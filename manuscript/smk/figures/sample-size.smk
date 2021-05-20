rule sample_size:
    input:
        lantern = lambda wildcards: (
            expand(
                "experiments/{ds}-{phenotype}/lantern/cv{cv}-n{n}/pred-val.csv",
                n=get(config, f"{wildcards.ds}/N", default=40000),
                cv=range(10),
                allow_missing=True,
            )
        ),
        feedforward = lambda wildcards: (
            expand(
                "experiments/{ds}-{phenotype}/feedforward-K8-D1-W32/cv{cv}-n{n}/pred-val.csv",
                n=get(config, f"{wildcards.ds}/N", default=40000),
                cv=range(10),
                allow_missing=True,
            )
        ),
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/sample-size-{p}.png"
    params:
        noiseless = (
            lambda wc: get(
                config,
                f"{wc.ds}/errors/{wc.phenotype}",
                default=["dummy-var"] # build a default list of errors if none there
                * len(get(config, f"{wc.ds}/phenotypes/{wc.phenotype}")),
            )[int(wc.p)]
            == "dummy-var" # noiseless if no errors
        )
    run:
        scores = None
        reg = re.compile("cv(?P<cv>\d+)-n(?P<n>\d+)")
        for inp in input.lantern:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(
                inp, i=int(wildcards.p), noiseless=params.noiseless
            ).assign(
                size=int(mtch.group("n")), cv=int(mtch.group("cv")), model="LANTERN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        for inp in input.feedforward:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(
                inp, i=int(wildcards.p), noiseless=params.noiseless
            ).assign(
                size=int(mtch.group("n")),
                cv=int(mtch.group("cv")),
                model="Feed-forward NN",
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        plot = (
            ggplot(scores, aes(x="size", y="metric", color="model"))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("observations")
            + theme(figure_size=(2, 2), dpi=300)
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)

rule laci_ss_ec50:
    input:
        lantern = (
            expand(
                "experiments/laci-joint/lantern/cv{cv}-n{n}/pred-val.csv",
                n=np.arange(5000, 45000, 5000),
                cv=range(10),
            )
        ),
        feedforward = (
            expand(
                "experiments/laci-joint/feedforward-K8-D1-W32/cv{cv}-n{n}/pred-val.csv",
                n=np.arange(5000, 45000, 5000),
                cv=range(10),
            )
        ),
    group: "figure"
    output:
        "figures/laci-joint/sample-size-ec50.png"
    run:

        scores = None
        reg = re.compile("cv(?P<cv>\d+)-n(?P<n>\d+)")
        for inp in input.lantern:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(inp).assign(
                size=int(mtch.group("n")), cv=int(mtch.group("cv")), model="LANTERN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        for inp in input.feedforward:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(inp).assign(
                size=int(mtch.group("n")), cv=int(mtch.group("cv")), model="Feed-forward NN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        plot = (
            ggplot(scores, aes(x="size", y="metric", color="model"))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("observations")
            + theme(figure_size=(2, 2), dpi=300)
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)

rule laci_ss_ginf:
    input:
        expand("experiments/laci-joint/lantern/cv{cv}-n{n}/pred-val.csv", n=np.arange(5000, 45000, 5000), cv=range(10)),
    group: "figure"
    output:
        "figures/laci-joint/sample-size-ginf.png"
    run:

        scores = None
        reg = re.compile("cv(?P<cv>\d+)-n(?P<n>\d+)")
        for inp in input:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(inp, i=1, noiseless=True).assign(
                size=int(mtch.group("n")), cv=int(mtch.group("cv")), model="LANTERN"
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        plot = (
            ggplot(scores, aes(x="size", y="metric", color="model"))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("observations")
            + theme(figure_size=(2, 2), dpi=300)
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
