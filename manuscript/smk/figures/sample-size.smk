
rule laci_ss:
    input:
        expand("experiments/laci-joint/lantern/cv{cv}-n{n}/pred-val.csv", n=np.arange(5000, 45000, 5000), cv=range(10)),
    output:
        "figures/laci-joint/sample-size.png"
    run:

        scores = None
        reg = re.compile("cv(?P<cv>\d+)-n(?P<n>\d+)")
        for inp in input[0]:
            mtch = reg.search(inp)
            _scores = src.predict.cv_scores(inp).assign(
                size=int(mtch.group("n")), cv=int(mtch.group("cv"), model="LANTERN")
            )

            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        (
            ggplot(scores, aes(x="size", y="metric", color="model"))
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("$R^2$")
            + xlab("observations")
            + theme(figure_size=(2, 2), dpi=300)
        )

