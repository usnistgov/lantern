
rule dimensions:
    input:
        expand(
            "experiments/{ds}/lantern/full/model.pt",
            ds=["gfp-brightness", "laci-joint", "covid-joint"],
        )
    output:
        "figures/dimensions.png"
    group: "figure"
    run:
        mn = 100
        mx = 1e-4

        def invgammalogpdf(x, alpha, beta):
            return alpha * beta.log() - torch.lgamma(alpha) + (-alpha-1)*x.log() - beta/x

        plt.figure(figsize=(2.5, 3), dpi=300)
        for pth in input:

            ds, phenotype = pth.split("/")[1].split("-")
            lab = config[ds]["label"]
            K = config["figures"]["effects"][f"{ds}-{phenotype}"]["zdim"]

            def dsget(pth, default):
                """Get the configuration for the specific dataset"""
                return get(config, f"{ds}/{pth}", default=default)

            df, ds, model = util.load_run(
                ds, phenotype, "lantern", "full", dsget("K", 8)
            )
            model.eval()

            X = ds[: len(ds)][0]

            with torch.no_grad():
                mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
                mu = mu[model.basis.order]

            plt.plot(np.arange(1, K + 1), mu[:K], marker="o", label=lab, zorder=10)

            mn = min(mn, mu[:K].min())
            mx = max(mx, mu[:K].max())

        plt.legend(
            handlelength=0.6,
        )
        plt.xlabel("dimension")
        plt.ylabel("$\sigma^2_k$")

        twy = plt.twiny()
        z = torch.logspace(np.log10(mn), np.log10(mx))
        plt.plot(
            invgammalogpdf(z, torch.tensor(0.001), torch.tensor(0.001)).exp().numpy(),
            z.numpy(),
            c="k",
            zorder=0,
        )
        plt.xlabel("prior probability")

        plt.semilogy()
        plt.savefig(output[0], bbox_inches="tight", verbose=False)

rule dimensions_percent:
    input:
        expand(
            "experiments/{ds}/lantern/full/model.pt",
            ds=["gfp-brightness", "laci-joint", "covid-joint"],
        )
    output:
        "figures/dimensions-percent.png"
    group: "figure"
    run:
        mn = 100
        mx = 1e-4

        def invgammalogpdf(x, alpha, beta):
            return alpha * beta.log() - torch.lgamma(alpha) + (-alpha-1)*x.log() - beta/x

        percents = []
        labels = []
        for pth in input:

            ds, phenotype = pth.split("/")[1].split("-")
            lab = config[ds]["label"]
            labels.append(lab)
            K = config["figures"]["effects"][f"{ds}-{phenotype}"]["zdim"]

            def dsget(pth, default):
                """Get the configuration for the specific dataset"""
                return get(config, f"{ds}/{pth}", default=default)

            df, ds, model = util.load_run(
                ds, phenotype, "lantern", "full", dsget("K", 8)
            )
            model.eval()

            X = ds[: len(ds)][0]

            with torch.no_grad():
                mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
                mu = mu[model.basis.order]

            percents.append(mu[:K]/mu[:K].sum())

            # plt.plot(np.arange(1, K + 1), mu[:K]/mu.sum(), marker="o", label=lab, zorder=10)

            mn = min(mn, (mu[:K]/mu[:K].sum()).min())
            mx = max(mx, (mu[:K]/mu[:K].sum()).max())

        plt.figure(figsize=(2, 2), dpi=200)
        lines = []
        for i, (p, lab) in enumerate(zip(percents, labels)):
            ax = plt.subplot(1, 3, i+1)
            K = len(p)
            (l,) = plt.plot(
                np.arange(1, K + 1), p, marker="o", label=lab, zorder=10, color=f"C{i}"
            )
            lines.append(l)
            plt.ylim(mn*0.6, 2)
            plt.xlim(0.2, K+0.8)
            plt.semilogy()

            if i > 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel("% total variance")

            if i == 1:
                plt.xlabel("dimensions")
            # ax.set_xticklabels([])
            plt.xticks([])

            # plt.legend()

        fig = plt.gcf()
        fig.legend(handles=lines, bbox_to_anchor=(.08, 0.92), loc="lower left")

        # plt.xlabel("dimension")
        # plt.ylabel("$\sigma^2_k$")

        plt.savefig(output[0], bbox_inches="tight")

rule dimensions_sim:
    input:
        "experiments/sim{ds}-phenotype/lantern/full/model.pt",
    output:
        "figures/sim{ds}-phenotype/dimensions.png"
    group: "figure"
    run:
        mn = 1e-4
        mx = 1

        def invgammalogpdf(x, alpha, beta):
            return alpha * beta.log() - torch.lgamma(alpha) + (-alpha-1)*x.log() - beta/x

        plt.figure(figsize=(2.5, 3), dpi=300)

        pth = input[0]
        ds, phenotype = pth.split("/")[1].split("-")
        lab = ds

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"sim{wildcards.ds}/{pth}", default=default)

        df, ds, model = util.load_run(ds, phenotype, "lantern", "full", dsget("K", 8))
        model.eval()

        X = ds[:len(ds)][0]

        with torch.no_grad():
            mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
            mu = mu[model.basis.order]

        plt.plot(np.arange(1, mu.shape[0]+1), mu, marker="o", label=lab, zorder=10)

        mn = min(mn, mu.min())
        mx = max(mx, mu.max())

        plt.legend(
            handlelength=0.6,
        )
        plt.xlabel("dimension")
        plt.ylabel("$\sigma^2_k$")

        twy = plt.twiny()
        z = torch.logspace(np.log10(mn), np.log10(mx))
        plt.plot(
            invgammalogpdf(z, torch.tensor(0.001), torch.tensor(0.001)).exp().numpy(),
            z.numpy(),
            c="k",
            zorder=0,
        )
        plt.xlabel("prior probability")

        plt.semilogy()
        plt.savefig(output[0], bbox_inches="tight", verbose=False)


rule dimensions_count_diagnostic:
    input:
        "experiments/{ds}/lantern/full/model.pt",
    output:
        "figures/{ds}/dimensions-count-diagnostic.png"
    group: "figure"
    run:
        mn = 1e-4
        mx = 1

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"simulations/{wildcards.ds}/{pth}", default=default)

        pth = input[0]
        ds, phenotype = pth.split("/")[1].split("-")
        df, ds, model = util.load_run(ds, phenotype, "lantern", "full", dsget("K", 8))
        model.eval()

        with torch.no_grad():
            mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
            mu = mu[model.basis.order]
            Wmu = model.basis.W_mu[:, model.basis.order].detach()
            Wstd = model.basis.W_log_sigma.exp()[:, model.basis.order].detach()
            qW = torch.distributions.Normal(Wmu, Wstd)

        
        alpha = torch.logspace(-4, -1)

        cdf = qW.cdf(torch.zeros_like(Wmu))
        cdf = torch.min(cdf, 1-cdf)

        cnt = torch.zeros(100, dsget("K", 8))
        for i, a in enumerate(alpha):
            c = (cdf < a/2).sum(axis=0)
            cnt[i, :] = c

        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(alpha.numpy(), cnt.numpy())
        plt.semilogx()
        plt.yscale("symlog", linthresh=(cnt[cnt>0]).min().item()*0.99);

        plt.title("Significantly non-zero effects")
        plt.ylabel("count")
        plt.xlabel("Significance level")

        plt.savefig(output[0], bbox_inches="tight", verbose=False)

rule dimensions_fold_change:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    group: "figure"
    output:
        "figures/{ds}-{phenotype}/dimensions_fold_change.png"
    group: "figure"
    run:

        import seaborn as sns

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/effects/{wildcards.ds}-{wildcards.phenotype}/{pth}",
                default=default,
            )
        from scipy.stats import norm

        K = dsget("K", 8)

        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", K
        )

        qalpha = model.basis.qalpha(detach=True)
        sigma = 1/qalpha.mean[model.basis.order]
        fold = (sigma[:-1] / sigma[1:]).numpy()[::-1]
        thresh = 1
        select = (np.where(np.log10(fold) > thresh)[0]).min()


        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(fold, marker="o")
        plt.scatter(range(select, K-1), fold[select:], facecolors="none", s=100, edgecolors="C0")
        plt.xticks(range(K-1), [f"z{K-k-1}" for k in range(K-1)])
        plt.ylabel("fold-change")
        plt.axhline(10**thresh, c="r", ls="--")

        plt.semilogy()
        plt.savefig(output[0], bbox_inches="tight", verbose=False)

rule dimensions_logprob:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt",
        "experiments/{ds}-{phenotype}/lantern/full/loss.pt",
    output:
        "figures/{ds}-{phenotype}/dimensions-logprob.png"
    group: "figure"
    resources:
        gres="gpu:1"
    run:

        import seaborn as sns
        from src import predict

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        def fget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(
                config,
                f"figures/dimensions/{wildcards.ds}-{wildcards.phenotype}/{pth}",
                default=default,
            )

        from scipy.stats import norm

        K = dsget("K", 8)
        maxDist = fget("maxDist", None)

        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", K
        )

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        loss.load_state_dict(torch.load(input[3]))

        model.eval()
        loss.eval()

        lp = src.predict.logprob_scan(
            ds.D,
            K,
            model,
            loss.losses[1].mll.likelihood,
            torch.utils.data.Subset(ds, np.where(df.distance <= maxDist)[0]) if maxDist is not None else ds,
            # torch.utils.data.Subset(ds, np.where(df["ginf-norm"] >= -1)[0]),
            pbar=True,
            cuda=torch.cuda.is_available(),
            size=1000,
        )

        useDims = fget("useDims", None)
        if useDims is not None:
            lp = lp.filter(regex=f"lp[{''.join(map(str, useDims))}].*")

        mean = [
            (
                lp.filter(regex=f".*k{k+1}", axis=1).sum(axis=1)
                - lp.filter(regex=f".*k{k}", axis=1).sum(axis=1)
            ).mean()
            for k in range(K)
        ]
        std = [
            2*(
                lp.filter(regex=f".*k{k+1}", axis=1).sum(axis=1)
                - lp.filter(regex=f".*k{k}", axis=1).sum(axis=1)
            ).std()
            / (lp.shape[0] ** 0.5)
            for k in range(K)
        ]

        fig, axes = plt.subplots(1, K, figsize=(K, 2))

        for k in range(K):
            axes[k].errorbar([0.05], mean[k], std[k], fmt=".", ecolor="k", markersize=10)
            ymn, ymx = axes[k].get_ylim()
            axes[k].set_ylim(min(0, ymn), max(ymx, 0)*1.2)
            axes[k].set_title(f"$z_{k+1}$")
            # axes[k].axhline(0, c="r")
            axes[k].set_xlim(-0.001, 0.101)
            axes[k].set_xticks([])
            axes[k].tick_params(axis="x", which="both", bottom=False)
            axes[k].spines.left.set_position('zero')
            axes[k].spines.right.set_color('none')
            axes[k].spines.bottom.set_position("zero")
            axes[k].spines.top.set_color('none')
            axes[k].xaxis.set_ticks_position('bottom')
            axes[k].yaxis.set_ticks_position('left')

        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3, 2))

        # ax1.spines["bottom"].set_visible(False)
        # ax1.tick_params(axis="x", which="both", bottom=False)
        # ax2.spines["top"].set_visible(False)

        # bs = max(mean[1:]) * 2
        # ts = mean[0] * 0.9

        # ymn, _ = ax2.get_ylim()
        # print(ymn)
        # ax2.set_ylim(-0.1, bs)
        # ax1.set_ylim(ts, mean[0]*1.1)
        # #ax1.set_yticks(np.arange(1000, 1501, 100))

        # # bars1 = ax1.bar(np.arange(K), mean, yerr=std)
        # # bars2 = ax2.bar(np.arange(K), mean, yerr=std)
        # err1 = ax1.errorbar(np.arange(K), mean, yerr=std, fmt=".", ecolor="k")
        # err2 = ax2.errorbar(np.arange(K), mean, yerr=std, fmt=".", ecolor="k")

        # d = 0.015
        # kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        # ax1.plot((-d, +d), (-d, +d), **kwargs)
        # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        # kwargs.update(transform=ax2.transAxes)
        # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # # for b1, b2 in zip(bars1, bars2):
        # #     posx = b2.get_x() + b2.get_width() / 2.0
        # #     if b2.get_height() > bs:
        # #         ax2.plot(
        # #             (posx - 3 * d, posx + 3 * d),
        # #             (1 - d, 1 + d),
        # #             color="k",
        # #             clip_on=False,
        # #             transform=ax2.get_xaxis_transform(),
        # #         )
        # #     if b1.get_height() > ts:
        # #         ax1.plot(
        # #             (posx - 3 * d, posx + 3 * d),
        # #             (-d, +d),
        # #             color="k",
        # #             clip_on=False,
        # #             transform=ax1.get_xaxis_transform(),
        # #         )

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", verbose=False)
