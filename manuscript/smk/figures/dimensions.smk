
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
            return get(config, f"simulations/{wildcards.ds}/{pth}", default=default)

        df, ds, model = util.load_run(ds, phenotype, "lantern", "full", 8)
        model.eval()

        X = ds[:len(ds)][0]

        with torch.no_grad():
            mu = (model.basis.log_beta.exp()) / (model.basis.log_alpha.exp() - 1)
            mu = mu[model.basis.order]

        plt.plot(np.arange(1, 9), mu, marker="o", label=lab, zorder=10)

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
