from glob import glob
import pandas as pd
from plotnine import *
from sklearn.metrics import r2_score, mean_squared_error

rule cvr2:
    """
    Cross-validated coefficient of determination (R^2) across models and datasets.
    """

    input:
        expand("experiments/{dataset}/{model}/cv{cv}/pred-val.csv", cv=range(10), model=TORCH_MODELS+["globalep"], dataset=DATASETS)
    group: "figure"
    output:
        "figures/cvr2.png"
    run:
        scores = None
        metric = r2_score
        dsets = []

        for dlabel, mlabel, ds, model, p, noiseless in [

                ("avGFP brightness", "Linear", "gfp-brightness", "linear", 0, True),
                ("avGFP brightness", "LANTERN", "gfp-brightness", "lantern", 0, True),
                ("avGFP brightness", "NN (K=1)", "gfp-brightness", "feedforward-K1-D1-W32", 0, True),
                ("avGFP brightness", "NN (K=8)", "gfp-brightness", "feedforward-K8-D1-W32", 0, True),
                ("avGFP brightness", "I-spline", "gfp-brightness", "globalep", 0, True),

                # ("SARS-CoV-2 $K_d$", "LANTERN", "covid-bind", "lantern", 0, False),
                ("SARS-CoV-2 $K_d$", "LANTERN", "covid-joint", "lantern", 1, False),
                ("SARS-CoV-2 $K_d$", "Linear", "covid-bind", "linear", 0, False),
                ("SARS-CoV-2 $K_d$", "NN (K=1)", "covid-bind", "feedforward-K1-D1-W32", 0, False),
                ("SARS-CoV-2 $K_d$", "NN (K=8)", "covid-bind", "feedforward-K8-D1-W32", 0, False),
                ("SARS-CoV-2 $K_d$", "I-spline", "covid-bind", "globalep", 0, False),

                # ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "LANTERN", "covid-exp", "lantern", 0, False),
                ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "LANTERN", "covid-joint", "lantern", 0, False),
                ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "Linear", "covid-exp", "linear", 0, False),
                ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "NN (K=1)", "covid-exp", "feedforward-K1-D1-W32", 0, False),
                ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "NN (K=8)", "covid-exp", "feedforward-K8-D1-W32", 0, False),
                ("SARS-CoV-2 $\log \Delta \mathrm{MFI}$", "I-spline", "covid-exp", "globalep", 0, False),

                # ("LacI $\mathrm{EC}_{50}$", "LANTERN", "laci-ec50", "lantern", 0, False),
                ("LacI $\mathrm{EC}_{50}$", "LANTERN", "laci-joint", "lantern", 0, False),
                ("LacI $\mathrm{EC}_{50}$", "Linear", "laci-ec50", "linear", 0, False),
                ("LacI $\mathrm{EC}_{50}$", "NN (K=1)", "laci-ec50", "feedforward-K1-D1-W32", 0, False),
                ("LacI $\mathrm{EC}_{50}$", "NN (K=8)", "laci-ec50", "feedforward-K8-D1-W32", 0, False),
                ("LacI $\mathrm{EC}_{50}$", "I-spline", "laci-ec50", "globalep", 0, False),

                # ("LacI $\mathrm{G}_{\infty}$", "LANTERN", "laci-ginf", "lantern", 0, True),
                ("LacI $\mathrm{G}_{\infty}$", "LANTERN", "laci-joint", "lantern", 1, True),
                ("LacI $\mathrm{G}_{\infty}$", "Linear", "laci-ginf", "linear", 0, True),
                ("LacI $\mathrm{G}_{\infty}$", "NN (K=1)", "laci-ginf", "feedforward-K1-D1-W32", 0, True),
                ("LacI $\mathrm{G}_{\infty}$", "NN (K=8)", "laci-ginf", "feedforward-K8-D1-W32", 0, True),
                ("LacI $\mathrm{G}_{\infty}$", "I-spline", "laci-ginf", "globalep", 0, True),

        ]:
            if dlabel not in dsets:
                dsets.append(dlabel)
                
            pths = expand(
                r"experiments/{dataset}/{model}/cv{cv}/pred-val.csv",
                cv=range(10),
                model=model,
                dataset=ds
            )

            _scores = pd.concat([pd.read_csv(pth) for pth in pths])
            if mlabel == "I-spline":
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
        
            _scores = _scores.assign(method=mlabel, dataset=dlabel)
            if scores is None:
                scores = _scores
            else:
                scores = pd.concat((scores, _scores))

        # enforce order we want
        scores = scores.assign(
            method=scores.method.astype("category").cat.reorder_categories(
                ["Linear", "I-spline", "NN (K=1)", "NN (K=8)", "LANTERN",]
            ),
            dataset=scores.dataset.astype("category").cat.reorder_categories(dsets),
        )

        # make the plot
        ncol = 3
        plot = (
            ggplot(scores)
            + aes(x="factor(method)", y="metric", fill="factor(method)")
            + geom_boxplot(outlier_alpha=0.0)
            + geom_jitter()
            + facet_wrap("dataset", scales="free_y", ncol=ncol)
            + theme(
                subplots_adjust={"wspace": 0.25},
                figure_size=(9, 6),
                dpi=300,
            )
            + guides(fill=guide_legend(title=""))
            + scale_x_discrete(name="", labels=[])
            + scale_y_continuous(name="$R^2$")
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
