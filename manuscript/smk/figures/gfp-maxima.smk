rule gfp_maxima_variants:
    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "output/gfp-brightness/local-optima-variants-{ind}.csv",
        "output/gfp-brightness/local-optima-mutations-{ind}.csv",
    run:
        from functools import reduce
        
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()

        X, y = ds[:len(ds)]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        Wmu = model.basis.W_mu[:, model.basis.order[:3]].detach()
        Wvar = model.basis.W_log_sigma[:, model.basis.order[:3]].exp().detach()

        counts = []

        i = int(wildcards.ind) - 1

        (c1, r1, c2, r2) = [(10.8, 0.5, -3.0, 0.6), (6.75, 0.25, 2.0, 0.2)][i]

        # start loop
        # look in ellipse
        (ind,) = torch.where(
            ((Z[:, 0] - c1) / r1) ** 2 + ((Z[:, 1] - c2) / r2) ** 2 < 1
        )

        df.iloc[ind, :].to_csv(output[0], index=False)

        variants = X[ind, :]
        wind, = torch.where(torch.any(variants, dim=0))

        counts.append(variants.sum(axis=1))

        Wopt = Wmu[wind, :]
        wsort = Wopt[:, 1].argsort(descending=True)

        wind = [wind[j] for j in wsort]
        toks = [ds.tokenizer.tokens[wi] for wi in wind]

        out = pd.DataFrame(Wopt[wsort, :].numpy())
        out["mutation"] = toks
        out.to_csv(output[1], index=False)

rule gfp_local_optima:
    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/local-optima-{ind}.png",
        "figures/gfp-brightness/local-optima-{ind}-count.png",
        # "figures/gfp-brightness/local-optima-1.png",
        # "figures/gfp-brightness/local-optima-2.png",
        #"figures/gfp-brightness/local-optima-counts.png",
    group: "figure"
    run:
        from functools import reduce
        
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()

        X, y = ds[:len(ds)]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        Wmu = model.basis.W_mu[:, model.basis.order[:3]].detach()
        Wvar = model.basis.W_log_sigma[:, model.basis.order[:3]].exp().detach()

        counts = []

        i = int(wildcards.ind) - 1

        (c1, r1, c2, r2) = [(10.8, 0.5, -3.0, 0.6), (6.75, 0.25, 2.0, 0.2)][i]

        # for i, (c1, r1, c2, r2) in enumerate(
        #     [(10.8, 0.5, -3.0, 0.6), (6.75, 0.25, 2.0, 0.2)]
        # ):

        # start loop
        # look in ellipse
        (ind,) = torch.where(
            ((Z[:, 0] - c1) / r1) ** 2 + ((Z[:, 1] - c2) / r2) ** 2 < 1
        )

        variants = X[ind, :]
        wind, = torch.where(torch.any(variants, dim=0))

        counts.append(variants.sum(axis=1))

        # variants = df.iloc[ind.numpy(), :].aaMutations.str.split(":").tolist()
        # toks = list(reduce(lambda x, y: x.union(set(y)), variants, set()))
        # wind = [ds.tokenizer.tokens.index(t) for t in toks]
        Wopt = Wmu[wind, :]
        wsort = Wopt[:, 1].argsort(descending=True)

        wind = [wind[j] for j in wsort]
        toks = [ds.tokenizer.tokens[wi] for wi in wind]

        W1 = 3
        W2 = 5
        H = 20
        SIZE = (H + 2, W1 + W2)
        fig = plt.figure(figsize=(6, 10))

        Wax = plt.subplot2grid(SIZE, (0, 0), colspan=W1, rowspan=H)
        im = Wax.imshow(
            Wopt[wsort, :].numpy(),
            aspect="auto",
            interpolation="none",
            origin="lower",
            cmap="PiYG",
            vmin=-Wopt[wsort, :].abs().max().item(),
            vmax=Wopt[wsort, :].abs().max().item(),
        )
        Wax.set_xticks([0, 1, 2])
        Wax.set_xticklabels([f"$z_{j+1}$" for j in range(3)])
        Wax.set_yticks(np.arange(Wopt.shape[0]))
        Wax.set_yticklabels(toks)

        Vax = plt.subplot2grid(SIZE, (0, W1), colspan=W2, rowspan=H)
        Vax.imshow(
            # variants[:, wind][:, wsort].t().numpy(),
            variants[:, wind].t().numpy(),
            aspect="auto",
            interpolation="none",
            origin="lower",
            cmap="Greys"
        )
        Vax.set_xticks([])
        Vax.set_yticks([])
        Vax.set_xlabel("variants")

        Cax = plt.subplot2grid(SIZE, (H+1, 0), colspan=W1 + W2)
        fig.colorbar(im, cax=Cax, orientation="horizontal")

        plt.savefig(output[0], bbox_inches="tight")

        out = pd.DataFrame(Wopt[wsort, :].numpy())
        out["mutation"] = toks
        out.to_csv(f"output/gfp-focus-w{i}.csv", index=False)
        # end loop

        # plt.figure(figsize=(3, 2), dpi=100)
        # plt.hist(X.sum(axis=1).numpy(), bins=15, log=True, density=True)
        # plt.hist(counts[0].numpy(), bins=5, log=True, density=True)
        # plt.hist(counts[1].numpy(), bins=5, log=True, density=True)
        # plt.savefig(output[-1], bbox_inches="tight")

        # plt.figure(figsize=(6, 3), dpi=200)
        # plt.subplot(121)
        plt.figure(figsize=(4, 6), dpi=200)
        plt.subplot(211)
        plt.scatter(Wopt[wsort, 1].numpy(), X[:, wind].sum(axis=0).numpy())
        plt.semilogy()
        plt.ylabel("Total observations")
        # plt.xlabel(r"$\mathrm{E}_q[w_{i2}]$")

        # plt.subplot(122)
        plt.subplot(212)
        plt.scatter(Wopt[wsort, 1].numpy(), Wvar[wind, 1].numpy())
        plt.semilogy()
        plt.ylabel(r"$\mathrm{Var}_q[w_{i2}]$")
        plt.xlabel(r"$\mathrm{E}_q[w_{i2}]$")

        plt.tight_layout()
        plt.savefig(output[1], bbox_inches="tight")

rule gfp_local_optima_count:
    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/local-optima-count.png",
    group: "figure"
    run:
        from functools import reduce
        
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()

        X, y = ds[:len(ds)]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        Wmu = model.basis.W_mu[:, model.basis.order[:3]].detach()
        Wvar = model.basis.W_log_sigma[:, model.basis.order[:3]].exp().detach()

        counts = []

        fig = plt.figure(figsize=(6, 3), dpi=200)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        for i, (c1, r1, c2, r2) in enumerate(
            [(10.8, 0.5, -3.0, 0.6), (6.75, 0.25, 2.0, 0.2)]
        ):

            # look in ellipse
            (ind,) = torch.where(
                ((Z[:, 0] - c1) / r1) ** 2 + ((Z[:, 1] - c2) / r2) ** 2 < 1
            )

            variants = X[ind, :]
            wind, = torch.where(torch.any(variants, dim=0))

            counts.append(variants.sum(axis=1))

            Wopt = Wmu[wind, :]
            wsort = Wopt[:, 1].argsort(descending=True)

            wind = [wind[j] for j in wsort]
            toks = [ds.tokenizer.tokens[wi] for wi in wind]

            ax1.scatter(Wopt[wsort, 1].numpy(), X[:, wind].sum(axis=0).numpy(), label=f"peak {i+1}")
            ax2.scatter(Wopt[wsort, 1].numpy(), Wvar[wind, 1].numpy())

        ax1.set_title("Total observations")
        ax1.set_xlabel(r"$\mathrm{E}_q[w_{i2}]$")
        ax1.semilogy()

        ax2.set_title(r"$\mathrm{Var}_q[w_{i2}]$")
        ax2.set_xlabel(r"$\mathrm{E}_q[w_{i2}]$")
        ax2.semilogy()

        plt.tight_layout()
        fig.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.9),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.savefig(output[0], bbox_inches="tight")

rule dimensions_ks_gfp_peak:
    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt",
        "experiments/gfp-brightness/lantern/full/loss.pt",
    output:
        "figures/gfp-brightness/dimensions-peak-ks-hist.png",
        "figures/gfp-brightness/dimensions-peak-ks.png"
    group: "figure"
    resources:
        gres="gpu:1"
    run:

        import seaborn as sns
        from src import predict

        wildcards.ds = "gfp"
        wildcards.phenotype = "brightness"
        wildcards.rerun = ""

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

        X, y = ds[:len(ds)]
        with torch.no_grad():
            Z = model.basis(X.cuda())
            Z = Z[:, model.basis.order].cpu()

        Wmu = model.basis.W_mu[:, model.basis.order[:3]].detach()
        Wvar = model.basis.W_log_sigma[:, model.basis.order[:3]].exp().detach()

        counts = []

        sel = None
        for i, (c1, r1, c2, r2) in enumerate(
            [(10.8, 0.5, -3.0, 0.6), (6.75, 0.25, 2.0, 0.2)]
        ):

            # look outside ellipse
            s = ((Z[:, 0] - c1) / r1) ** 2 + ((Z[:, 1] - c2) / r2) ** 2 > 1

            # combine
            if sel is None:
                sel = s
            else:
                sel = s & sel

        (ind,) = torch.where(sel)
        print(ind.shape, lp.shape)

        lp = lp.iloc[ind, :]

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

rule gfp_maxima_mutations:
    input:
        "output/gfp-brightness/local-optima-variants-{ind}.csv",
        "output/gfp-brightness/local-optima-mutations-{ind}.csv",
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt",
    output:
        "figures/gfp-brightness/local-maxima-{ind}-mutations.png"
    run:
        variants = pd.read_csv(input[0], index_col=0)
        mutations = pd.read_csv(input[1])

        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)

        if wildcards.ind == "1":
            muts = mutations.iloc[-15:].iloc[::-1, :]
        else:
            muts = mutations.iloc[:15]

        fig = plt.figure(figsize=(6, 8))
        for i, (_, m) in enumerate(muts.iterrows()):
            plt.subplot(5, 3, i + 1)

            tmp = df.loc[
                df.substitutions.str.contains(m.mutation).replace(np.nan, False), :
            ]

            # remove variants in maxima
            for _, v in variants.loc[
                variants.substitutions.str.contains(m.mutation), :
            ].iterrows():
                tmp = tmp.loc[tmp.index != v.name]
            
            plt.hist2d(
                tmp.substitutions.replace(np.nan, "").str.split(":").apply(len),
                tmp.medianBrightness,
                cmap="Blues",
                vmin=0,
                cmin=1,
            )
            plt.colorbar()

            for _, v in variants.loc[
                variants.substitutions.str.contains(m.mutation), :
            ].iterrows():
                plt.scatter(
                    len(v.substitutions.split(":")),
                    v.medianBrightness,
                    color="red",
                    marker="*",
                    s = 25
                )
                yl = plt.ylim()
                plt.ylim(yl[0], max(yl[1], v.medianBrightness + 0.2))
                xl = plt.xlim()
                plt.xlim(xl[0], max(xl[1], len(v.substitutions.split(":")) + 0.2))

            plt.title(m.mutation)
            if i > 11:
                plt.xlabel("# of mutations")
            if i % 3 == 0:
                plt.ylabel("median brightness")

        plt.tight_layout()
        print("saving...")
        plt.savefig(output[0])
