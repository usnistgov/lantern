
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

rule dimensions_percent_allostery:
    input:
        expand(
            "experiments/allostery{dim}-joint/lantern/full/model.pt",
            dim=range(1, 4)
        )
    output:
        "figures/dimensions-percent-allostery.png"
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

        plt.figure(figsize=(2.5, 3), dpi=300)
        # _, ax = plt.subplots(figsize=(2, 3), dpi=300)
        lines = []
        for i, (p, lab) in enumerate(zip(percents, labels)):
            # ax = plt.subplot(1, 3, i+1)
            ax = plt.subplot2grid((1, 6), (0, [0, 1, 3][i]), colspan=[1, 2, 3][i])
            K = len(p)
            l = plt.bar(np.arange(1, K + 1), p, log=True, label=lab, color=f"C{i}")
            lines.append(l)
            plt.ylim(mn*0.6, 2)
            plt.xlim(0.2, K+0.8)

            if i > 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel("% total variance")

            if i == 1:
                plt.xlabel("dimensions")
            plt.xticks(np.arange(1, K + 1), [f"$z_{d+1}$" for d in range(K)])

        plt.tight_layout()
        fig = plt.gcf()
        # fig.legend(handles=lines, bbox_to_anchor=(.08, 0.92), loc="lower left")
        fig.legend(
            bbox_to_anchor=(0.98, 0.86),
            loc="upper left",
            borderaxespad=0.0,
        )

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

rule dimensions_ks:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full{rerun,(-r.*)?}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/full{rerun,(-r.*)?}/loss.pt",
    output:
        "figures/{ds}-{phenotype}/dimensions-ks-hist{rerun,(-r.*)?}.png",
        "figures/{ds}-{phenotype}/dimensions-ks{rerun,(-r.*)?}.png"
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
        from scipy.stats import ks_2samp as ks2

        K = dsget("K", 8)
        maxDist = fget("maxDist", None)

        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", K, slug=wildcards.rerun
        )

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        loss.load_state_dict(torch.load(input[3]))

        # reload, b/c sometimes `loss` overrides
        df, ds, model = util.load_run(
            wildcards.ds, wildcards.phenotype, "lantern", "full", K, slug=wildcards.rerun
        )

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
            resample=1
        )

        stat = [
            ks2(
                lp.filter(regex=f".*k{k+1}$", axis=1).sum(axis=1),
                lp.filter(regex=f".*k{k}$", axis=1).sum(axis=1),
            )
            for k in range(K)
        ]

        def latex_float(f):
            float_str = "{0:.4g}".format(f)
            if "e" in float_str:
                base, exponent = float_str.split("e")
                return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
            else:
                return float_str

        fig, axes = plt.subplots(2, K//2, figsize=(K, 4))
        axes = axes.ravel()

        for k in range(K):
            axes[k].hist(
                lp.filter(regex=f".*k{k}", axis=1).sum(axis=1),
                label=f"K={k}",
                alpha=0.6,
                bins=50,
                log=True,
            )
            axes[k].hist(
                lp.filter(regex=f".*k{k+1}", axis=1).sum(axis=1),
                label=f"K={k+1}",
                alpha=0.6,
                bins=50,
                log=True,
            )
            axes[k].legend(shadow=True, fancybox=True)
            if k >= K//2:
                axes[k].set_xlabel("$E_q[\log p(y)]$")
            if (k % (K // 2)) == 0:
                axes[k].set_ylabel("count")
            # axes[k].set_title(f"p = {stat[k].pvalue:.4e}")
            axes[k].set_title(f"$p = {latex_float(stat[k].pvalue)}$")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", verbose=False, dpi=300)

        plt.figure(figsize=(3, 3))
        for r in range(2):
            plt.subplot(2, 1, r+1)
            plt.bar(np.arange(K), [s.pvalue for s in stat], log=(r == 1))
            plt.axhline(0.05, c="r")
            plt.ylabel("pvalue")
            plt.xticks([])

        plt.xticks(np.arange(K), [f"$z_{k+1}$" for k in range(K)])

        plt.tight_layout()
        plt.savefig(output[1], bbox_inches="tight", verbose=False, dpi=300)
