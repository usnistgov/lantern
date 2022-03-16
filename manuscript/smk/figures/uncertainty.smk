
rule uncertainty:
    input:
        expand(
            "experiments/{ds}/lantern/cv{c}/pred-val.csv",
            ds=["laci-joint", "covid-joint", "gfp-brightness",],
            c=range(10),
        )
    output:
        "figures/rmse-vs-dist.png",
        "figures/uncertainty-vs-rmse.png",
    run:
        def loadWithDist(pth, col, sigmay=None):
            df = pd.read_csv(pth)
            df["distance"] = df[col].replace(np.nan, "").str.split(":").apply(len)

            # add observation uncertainty
            if sigmay is not None:
                for d in range(len(sigmay)):
                    # print(sigmay[d])
                    df[f"yhat_std{d}"] = np.sqrt(df[f"yhat_std{d}"]**2 + sigmay[d])

            return df

        # load noise values
        noise = {}
        for _ds, _phen in [("laci", "joint"), ("gfp", "brightness"), ("covid", "joint")]:
            df, ds, model = util.load_run(_ds, _phen, "lantern", "full", 8,)
            loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
            D = len(get(config, f"{_ds}/phenotypes/{_phen}"))
            phenotypes = get(config, f"{_ds}/phenotype_labels")

            tmp = []
            with torch.no_grad():
                for pt in [
                    f"experiments/{_ds}-{_phen}/lantern/cv{c}/loss.pt"
                    for c in range(10)
                ]:
                    loss.load_state_dict(torch.load(pt, "cpu"))
                    likelihood = loss.losses[1].mll.likelihood

                    if hasattr(likelihood, "task_noises"):
                        tmp.append((likelihood.noise + likelihood.task_noises).numpy())
                    else:
                        tmp.append(likelihood.noise.numpy())

            noise[_ds] = tmp


        uncertainty = pd.concat(
            [
                pd.melt(
                    loadWithDist(
                        f"experiments/laci-joint/lantern/cv{c}/pred-val.csv",
                        "substitutions",
                        # noise["laci"][c],
                    )
                    .rename(
                        columns={
                            "yhat_std0": "LacI $\mathrm{EC}_{50}$",
                            "yhat_std1": "LacI $\mathrm{G}_{\infty}$",
                        }
                    )
                    .assign(dataset="LacI"),
                    ["distance", "dataset",],
                    ["LacI $\mathrm{EC}_{50}$", "LacI $\mathrm{G}_{\infty}$"],
                    var_name="target",
                    value_name="uncertainty",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    loadWithDist(
                        f"experiments/gfp-brightness/lantern/cv{c}/pred-val.csv",
                        "aaMutations",
                        # noise["gfp"][c],
                    )
                    .rename(
                        columns={
                            "n_aa_substitutions": "distance",
                            "yhat_std0": "avGFP brightness",
                        }
                    )
                    .assign(dataset="avGFP"),
                    ["distance", "dataset",],
                    ["avGFP brightness"],
                    var_name="target",
                    value_name="uncertainty",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    loadWithDist(
                        f"experiments/covid-joint/lantern/cv{c}/pred-val.csv",
                        "aa_substitutions",
                        # noise["covid"][c],
                    )
                    .rename(
                        columns={
                            # "n_aa_substitutions": "distance",
                            "yhat_std0": "SARS-Cov2 expression",
                            "yhat_std1": "SARS-Cov2 binding",
                        }
                    )
                    .assign(dataset="SARS-Cov2"),
                    ["distance", "dataset",],
                    ["SARS-Cov2 expression", "SARS-Cov2 binding"],
                    var_name="target",
                    value_name="uncertainty",
                )
                for c in range(10)
            ]
        )

        measure = pd.concat(
            [
                pd.melt(
                    loadWithDist(
                        f"experiments/laci-joint/lantern/cv{c}/pred-val.csv",
                        "substitutions",
                    )
                    .rename(
                        columns={
                            "y0": "LacI $\mathrm{EC}_{50}$",
                            "y1": "LacI $\mathrm{G}_{\infty}$",
                        }
                    )
                    .assign(dataset="LacI"),
                    ["distance", "dataset",],
                    ["LacI $\mathrm{EC}_{50}$", "LacI $\mathrm{G}_{\infty}$"],
                    var_name="target",
                    value_name="measurement",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    loadWithDist(
                        f"experiments/gfp-brightness/lantern/cv{c}/pred-val.csv",
                        "aaMutations",
                    )
                    .rename(
                        columns={
                            "n_aa_substitutions": "distance",
                            "y0": "avGFP brightness",
                        }
                    )
                    .assign(dataset="avGFP"),
                    ["distance", "dataset",],
                    ["avGFP brightness"],
                    var_name="target",
                    value_name="measurement",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    pd.read_csv(f"experiments/covid-joint/lantern/cv{c}/pred-val.csv")
                    .rename(
                        columns={
                            "n_aa_substitutions": "distance",
                            "y0": "SARS-Cov2 expression",
                            "y1": "SARS-Cov2 binding",
                        }
                    )
                    .assign(dataset="SARS-Cov2"),
                    ["distance", "dataset",],
                    ["SARS-Cov2 expression", "SARS-Cov2 binding"],
                    var_name="target",
                    value_name="measurement",
                )
                for c in range(10)
            ]
        )

        pred = pd.concat(
            [
                pd.melt(
                    loadWithDist(
                        f"experiments/laci-joint/lantern/cv{c}/pred-val.csv",
                        "substitutions",
                    )
                    .rename(
                        columns={
                            "yhat0": "LacI $\mathrm{EC}_{50}$",
                            "yhat1": "LacI $\mathrm{G}_{\infty}$",
                        }
                    )
                    .assign(dataset="LacI"),
                    ["distance", "dataset",],
                    ["LacI $\mathrm{EC}_{50}$", "LacI $\mathrm{G}_{\infty}$"],
                    var_name="target",
                    value_name="prediction",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    loadWithDist(
                        f"experiments/gfp-brightness/lantern/cv{c}/pred-val.csv",
                        "aaMutations",
                    )
                    .rename(
                        columns={
                            "n_aa_substitutions": "distance",
                            "yhat0": "avGFP brightness",
                        }
                    )
                    .assign(dataset="avGFP"),
                    ["distance", "dataset",],
                    ["avGFP brightness"],
                    var_name="target",
                    value_name="prediction",
                )
                for c in range(10)
            ]
            + [
                pd.melt(
                    pd.read_csv(f"experiments/covid-joint/lantern/cv{c}/pred-val.csv")
                    .rename(
                        columns={
                            "n_aa_substitutions": "distance",
                            "yhat0": "SARS-Cov2 expression",
                            "yhat1": "SARS-Cov2 binding",
                        }
                    )
                    .assign(dataset="SARS-Cov2"),
                    ["distance", "dataset",],
                    ["SARS-Cov2 expression", "SARS-Cov2 binding"],
                    var_name="target",
                    value_name="prediction",
                )
                for c in range(10)
            ]
        )

        full = uncertainty.copy()
        full["measurement"] = measure["measurement"]
        full["prediction"] = pred["prediction"]
        full["error"] = abs(full.measurement - full.prediction)
        full["error2"] = (full.measurement - full.prediction) ** 2

        # first figure
        from plotnine.stats.stat_summary import bootstrap_statistics
        from functools import partial

        rmse = partial(
            bootstrap_statistics,
            statistic=lambda x, axis=None: np.sqrt(np.mean(x, axis=axis)),
            n_samples=1000,
            confidence_interval=0.95,
            random_state=None,
        )

        plot = (
            ggplot(
                full[(full.distance > 0) & (full.distance < 7)],
                aes(x="distance", y="error2", color="target"),
            )
            + stat_summary(fun_data=rmse)
            + stat_summary(geom=geom_line, fun_data=rmse)
            + ylab("RMSE")
            + xlab("Mutational distance")
            + theme_matplotlib()
            + theme(figure_size=(3, 2), dpi=300)
        )

        plot.save(output[0], bbox_inches="tight", verbose=False)
        plt.close()

        # second figure
        grp = full[(full.distance > 0) & (full.distance < 7)].groupby(
            ["distance", "target"]
        )

        err = grp.apply(lambda x: np.sqrt(x.error2.mean())).reset_index(name="rmse")
        unc = grp.apply(lambda x: x.uncertainty.mean()).reset_index(name="uncertainty")

        mrg = pd.merge(err, unc)

        plot = (
            ggplot(mrg, aes(x="uncertainty", y="rmse", color="target"),)
            + stat_summary()
            + stat_summary(geom=geom_line)
            + ylab("RMSE")
            + xlab(r"$\sigma[y^* \vert x^*]$")
            + theme_matplotlib()
            + theme(figure_size=(3, 2), dpi=300)
        )
        plot.save(output[1], bbox_inches="tight", verbose=False)
