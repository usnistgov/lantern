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

rule calibration_interval:
    """
    Calibration of 95% credible prediction interval
    """

    input:
        predictions = expand(
            "experiments/{dataset}-{phenotype}/lantern/cv{cv}/pred-val.csv",
            cv=range(10),
            allow_missing=True,
        ),
        losses = expand(
            "experiments/{dataset}-{phenotype}/lantern/cv{cv}/loss.pt",
            cv=range(10),
            allow_missing=True,
        ),
    output:
        "figures/{dataset}-{phenotype}/cv-prediction-interval-calibration{norm,(-norm)?}.png"
    group: "figure"
    run:
        from scipy.stats import norm

        import src.calibration
        import util

        df, ds, model = util.load_run(wildcards.dataset, wildcards.phenotype, "lantern", "full", 8,)
        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        D = len(get(config, f"{wildcards.dataset}/phenotypes/{wildcards.phenotype}"))
        phenotypes = get(config, f"{wildcards.dataset}/phenotype_labels")

        noises = []
        with torch.no_grad():
            for pt in input.losses:
                loss.load_state_dict(torch.load(pt, "cpu"))
                likelihood = loss.losses[1].mll.likelihood

                if hasattr(likelihood, "task_noises"):
                    noises.append((likelihood.noise + likelihood.task_noises).numpy())
                else:
                    noises.append(likelihood.noise.numpy())

        crits = np.linspace(0.05, 0.95, 20)

        scores = None
        for pth, nz in zip(input.predictions, noises):
            tmp = pd.read_csv(pth)

            ydist = norm(
                tmp[[f"yhat{d}" for d in range(D)]],
                (tmp[[f"yhat_std{d}" for d in range(D)]]**2 + nz[None, :])**0.5,
            )

            for cc, c in enumerate(crits):
                tmp[[f"y{d}-interval{cc}" for d in range(D)]] = (
                    ydist.ppf(c / 2) < tmp[[f"y{d}" for d in range(D)]]
                ) & (tmp[[f"y{d}" for d in range(D)]] < ydist.ppf(1 - c / 2))

            # find weights if necessary
            weights = np.ones((tmp.shape[0], D)) / tmp.shape[0]
            if wildcards.norm != "":
                for d in range(D):
                    _, yweight = src.calibration.balanced_sample(tmp[f"y{d}"])
                    weights[:, d] = yweight

            # get to right shape
            weights = weights.repeat(crits.shape[0], axis=1)

            # take average of coverage
            tmp = (
                (
                    tmp[
                        [
                            f"y{d}-interval{cc}"
                            for d in range(D)
                            for (cc, _) in enumerate(crits)
                        ]
                    ]
                    * weights
                )
                .div(weights.sum(axis=0))
                .sum(axis=0)
                .to_frame(pth)
                .transpose()
                # .assign(path=pth)
            )

            if scores is None:
                scores = tmp
            else:
                scores = pd.concat((scores, tmp))

        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(crits, crits, c="k")
        for d in range(D):
            plt.plot(
                1-crits,
                scores[[f"y{d}-interval{cc}" for cc, _ in enumerate(crits)]].mean(),
                "o-",
                markersize=3,
                label=phenotypes[d],
            )

        plt.xlabel("expected coverage")
        plt.ylabel("observed coverage")

        plt.legend()

        plt.savefig(output[0], bbox_inches="tight")

rule calibration_interval_diagnostic:
    """
    95% credible prediction interval diagnostic
    """

    input:
        predictions = expand(
            "experiments/{dataset}-{phenotype}/lantern/cv{cv}/pred-val.csv",
            cv=range(1),
            allow_missing=True,
        ),
        losses = expand(
            "experiments/{dataset}-{phenotype}/lantern/cv{cv}/loss.pt",
            cv=range(1),
            allow_missing=True,
        )
    output:
        "figures/{dataset}-{phenotype}/cv-prediction-interval-calibration-diagnostic.png"
    group: "figure"
    run:
        from scipy.stats import norm
        import util
        import src.calibration

        # phenotypes = get(config, f"{wildcards.dataset}/phenotypes/{wildcards.phenotype}")
        phenotypes = list(
            get(
                config, f"figures/surface/{wildcards.dataset}-{wildcards.phenotype}"
            ).keys()
        )
        D = len(phenotypes)

        df, ds, model = util.load_run(wildcards.dataset, wildcards.phenotype, "lantern", "full", 8,)
        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        noises = src.calibration.noises(loss, input.losses)

        df = pd.read_csv(input.predictions[0])
        ypred = src.calibration.predictive_distribution(
            D, input.predictions[0], noises[0]
        )

        fig = plt.figure(figsize=(3*D, 6), dpi=300,)

        axes = []
        maps = []
        for d in range(D):

            target = phenotypes[d]
            raw = get(
                config,
                f"figures/surface/{wildcards.dataset}-{wildcards.phenotype}/{target}/raw",
                default=None,
            )

            mu = df[raw].mean() if raw is not None else 0
            std = df[raw].std() if raw is not None else 1

            # balanced sampling
            ax = plt.subplot(2, D, d+D+1)
            # ax = plt.subplot(2, 2, d + 1)
            axes.append(ax)
            ind, yweight, ycount = src.calibration.balanced_sample(df[f"y{d}"]*std + mu)

            # values
            y = df[f"y{d}"].values[ind]*std + mu
            ymu = ypred.mean()[ind, d]*std + mu
            yrange = ypred.std()[ind, d] * 2*std

            # colros
            norm = mpl.colors.LogNorm(
                # vmin=(1-yweight[ind]).min(), vmax=(1-yweight[ind]).max(), clip=True
                vmin=1, vmax=ycount.max(), clip=True
            )
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
            color = np.array([(mapper.to_rgba(v)) for v in ycount[ind]])
            maps.append(mapper)

            # plot comparison to observed and predictive interval
            for i in range(y.shape[0]):
                plt.errorbar(
                    y[i], ymu[i], yrange[i], ls="None", marker="o", color=color[i, :]
                )

            mn = (df[f"y{d}"]*std + mu).min()
            mx = (df[f"y{d}"]*std + mu).max()
            rng = mx - mn
            plt.plot([mn - 0.1*rng, mx + 0.1*rng], [mn - 0.1*rng, mx + 0.1*rng], c="r")
            if d == 0:
                plt.ylabel("predicted")
            plt.xlabel("observed")

            # unbalanced sampling
            # ax = plt.subplot(2, 2, d + 3)
            # ax = plt.subplot(2, 2, d + 1)
            ax = plt.subplot(2, D, d + 1)

            axes.append(ax)
            ind = np.random.choice(np.arange(df.shape[0]), 100, replace=False)

            # values
            y = df[f"y{d}"].values[ind]*std + mu
            ymu = ypred.mean()[ind, d] * std + mu
            yrange = ypred.std()[ind, d] * 2 * std

            # colros
            norm = mpl.colors.LogNorm(
                # vmin=(1-yweight[ind]).min(), vmax=(1-yweight[ind]).max(), clip=True
                vmin=1, vmax=ycount.max(), clip=True
            )
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
            color = np.array([(mapper.to_rgba(v)) for v in ycount[ind]])
            maps.append(mapper)

            # plot comparison to observed and predictive interval
            for i in range(y.shape[0]):
                plt.errorbar(
                    y[i], ymu[i], yrange[i], ls="None", marker="o", color=color[i, :]
                )

            mn = (df[f"y{d}"] * std + mu).min()
            mx = (df[f"y{d}"] * std + mu).max()
            rng = mx - mn
            plt.plot([mn - 0.1*rng, mx + 0.1*rng], [mn - 0.1*rng, mx + 0.1*rng], c="r")
            if d == 0:
                plt.ylabel("predicted")

        for d in range(D):
            cb = fig.colorbar(
                maps[d * 2],
                ax=[axes[2 * d], axes[2 * d + 1]],
                location="bottom",
                pad=0.125,
                label="# of similar phenotypes"
            )

        # fig.colorbar(maps[1], ax=[axes[2], axes[3]], location="bottom", pad=0.3)
        # fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.2)
        plt.savefig(output[0], bbox_inches="tight")
