import pandas as pd
from plotnine import *
from sklearn.metrics import r2_score, mean_squared_error

# subworkflow gfp_lantern:
#     snakefile:
#         "lantern.smk"
#     configfile:
#         "config/gfp.yaml"
# 
# subworkflow gfp_ff:
#     snakefile:
#         "feedforward.smk"
#     configfile:
#         "config/gfp.yaml"
# 
# subworkflow gfp_globalep:
#     snakefile:
#         "globalep.smk"
#     configfile:
#         "config/gfp.yaml"

rule cvr2:
    """
    Cross-validated coefficient of determination (R^2) across models and datasets.
    """

    input:
        gfp_lantern=gfp_lantern(expand("experiments/gfp/lantern/cv{cv}/pred-val.csv", cv=range(10))),
        gfp_ff_k1=gfp_lantern(expand("experiments/gfp/feedforward-K1/cv{cv}/pred-val.csv", cv=range(10))),
        gfp_ff_k8=gfp_lantern(expand("experiments/gfp/feedforward-K8/cv{cv}/pred-val.csv", cv=range(10))),
        gfp_globalep=gfp_globalep(expand("experiments/gfp-brightness/globalep/cv{cv}/pred-val.csv", cv=range(10))),
    output:
        "figures/cvr2.png"
    run:
        scores = None
        metric = r2_score

        for ds, model, pths, p, noiseless in [
                ("avGFP", "lantern", input.gfp_lantern, 0, True),
                ("avGFP", "NN (K=1)", input.gfp_ff_k1, 0, True),
                ("avGFP", "NN (K=8)", input.gfp_ff_k8, 0, True),
                ("avGFP", "I-spline", input.gfp_globalep, 0, True),
        ]:
            _scores = pd.concat([pd.read_csv(pth) for pth in pths])
            if model == "I-spline":
                print((_scores.groupby("cv")
                .apply(
                    lambda x: metric(
                        x.observed_phenotype,
                        x.func_score,
                        sample_weight=None if noiseless else 1 / x.func_score_var,
                    )
                )).head())

                _scores = (
                    _scores.groupby("cv")
                    .apply(
                        lambda x: metric(
                            x.observed_phenotype,
                            x.func_score,
                            sample_weight=None if noiseless else 1 / x.func_score_var,
                        )
                    )
                    .to_frame("metric")
                )
            else:
                _scores = (
                    _scores.groupby("cv")
                    .apply(
                        lambda x: metric(
                            x[f"y{p}"],
                            x[f"yhat{p}"],
                            sample_weight=None if noiseless else 1 / x[f"noise{p}"],
                        )
                    )
                    .to_frame("metric")
                )
        
            _scores = _scores.assign(method=model, dataset=ds)
            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))
        
        # enforce order we want
        scores = scores.assign(
            method=scores.method.astype("category").cat.reorder_categories([
                "I-spline",
                "NN (K=1)",
                "NN (K=8)",
                "lantern",
            ]),
            # dataset=scores.dataset.astype("category").cat.reorder_categories(dsets),
        )

        # make the plot
        ncol = 1
        plot = (
            ggplot(scores)
            + aes(x="factor(method)", y="metric", fill="factor(method)")
            + geom_boxplot()
            + geom_jitter()
            + facet_wrap("dataset", scales="free_y", ncol=ncol)
            + theme(
                subplots_adjust={"wspace": 0.45},
                figure_size=(3, 3),
                dpi=300,
            )
            + guides(fill=guide_legend(title=""))
            + scale_x_discrete(name="", labels=[])
            + scale_y_continuous(name="$R^2$")
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
