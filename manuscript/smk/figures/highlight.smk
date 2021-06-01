rule gfp_surface_focus:
    """
    Surface plot of lantern model.
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/brightness/surface-focus.png"
    run:
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()
        raw = "medianBrightness"

        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=0,
            alpha=0.001,
            lim = [-2.1, 1.1, -0.48, 0.22]
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=False,
            image=False,
            mask=False,
        )

        plt.savefig(output[0], bbox_inches="tight")

rule gfp_surface_bfp1:
    """
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/brightness/surface-bfp1.png"
    run:
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()
        raw = "medianBrightness"

        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=0,
            alpha=0.001,
            lim = [-1.1, 2.6, -0.88, 0.22]
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=False,
            image=False,
            mask=False,
        )

        X, y = ds[: len(ds)][:2]

        # get variant info
        w_mu = model.basis.W_mu.detach()[:, model.basis.order].numpy()

        wt = np.zeros(w_mu.shape[1])

        # substitutions
        subs = "SY64H".split(":")
        labels = [s[1:] for s in subs]
        ind = [ds.tokenizer.tokens.index(s) for s in subs]

        # from wild-type
        for i in ind:
            _z = w_mu[i, :]
            plt.arrow(
                wt[0],
                wt[1],
                _z[0],
                _z[1],
                color="C{}".format(ind.index(i)),
                label="+{}".format(labels[ind.index(i)]),
                length_includes_head=True,
                width=0.01,
            )

        fig.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.9),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.savefig(output[0], bbox_inches="tight")

rule gfp_surface_bfp2:
    """
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/brightness/surface-bfp2.png"
    run:
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()
        raw = "medianBrightness"

        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=0,
            alpha=0.001,
            lim = [-1.1, 2.6, -0.88, 0.22]
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=False,
            image=False,
            mask=False,
        )

        X, y = ds[: len(ds)][:2]

        # get variant info
        w_mu = model.basis.W_mu.detach()[:, model.basis.order].numpy()

        wt = np.zeros(w_mu.shape[1])

        # substitutions
        subs = ["SY143F", "SS63T", "SH229L",'SY37N', 'SN103T', 'SY143F', 'SI169V', 'SN196S', 'SA204V']
        labels = [s[1:] for s in subs]
        ind = [ds.tokenizer.tokens.index(s) for s in subs]

        # from wild-type
        for i in ind:
            _z = w_mu[i, :]
            plt.arrow(
                wt[0],
                wt[1],
                _z[0],
                _z[1],
                color="C{}".format(ind.index(i)),
                label="+{}".format(labels[ind.index(i)]),
                length_includes_head=True,
                width=0.01,
            )

        fig.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.9),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.savefig(output[0], bbox_inches="tight")

rule laci_parametric:
    """
    Parametric plot of LacI allosteric surface
    """

    input:
        "data/processed/laci.csv",
        "data/processed/laci-joint.pkl",
        "experiments/laci-joint/lantern/full/model.pt"
    output:
        "figures/laci-joint/parametric.png"
    run:
        df, ds, model = util.load_run("laci", "joint", "lantern", "full", 8)
        model.eval()

        plt.figure(figsize=(5, 3), dpi=200)

        plt.subplot(121)
        plt.plot(
            allostery.ec50(
                K_I=np.logspace(
                    np.log10(allostery.K_I_0 * 0.1), np.log10(allostery.K_I_0 * 10), 100
                )
            ),
            allostery.ginf(
                K_I=np.logspace(
                    np.log10(allostery.K_I_0 * 0.1), np.log10(allostery.K_I_0 * 10), 100
                )
            ),
            label=r"$\log(K_I)$",
        )

        plt.plot(
            allostery.ec50(
                delta_eps_AI=np.linspace(
                    allostery.delta_eps_AI_0 - 2 * np.log(10), allostery.delta_eps_AI_0 + 8 * np.log(10), 100
                )
            )
            + 0.2,
            allostery.ginf(
                delta_eps_AI=np.linspace(
                    allostery.delta_eps_AI_0 - 2 * np.log(10), allostery.delta_eps_AI_0 + 8 * np.log(10), 100
                )
            )
            + 0.2,
            label=r"$\Delta \epsilon_{AI}$",
        )

        plt.plot(
            allostery.ec50(
                delta_eps_RA=np.linspace(
                    allostery.delta_eps_RA_0 - 1.1 * np.log(10), allostery.delta_eps_RA_0 + 1.8 * np.log(10), 100
                )
            )
            - 0.2,
            allostery.ginf(
                delta_eps_RA=np.linspace(
                    allostery.delta_eps_RA_0 - 1.1 * np.log(10), allostery.delta_eps_RA_0 + 1.8 * np.log(10), 100
                )
            )
            - 0.2,
            label=r"$\Delta \epsilon_{RA}$",
        )

        plt.plot(
            allostery.ec50(
                K_A=np.logspace(
                    np.log10(allostery.K_A_0 * 0.1),
                    np.log10(allostery.K_A_0 * 100),
                    100,
                )
            ),
            allostery.ginf(
                K_A=np.logspace(
                    np.log10(allostery.K_A_0 * 0.1),
                    np.log10(allostery.K_A_0 * 100),
                    100,
                )
            ),
            label=r"$\log(K_A)$",
        )

        plt.scatter(allostery.ec50(), allostery.ginf(), c="k", zorder=100)

        plt.semilogx()
        plt.semilogy()

        plt.xlabel(r"$\mathrm{EC}_{50}$")
        plt.ylabel(r"$\mathrm{G}_{\infty}$")

        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
            handlelength=0.8
        )

        ax = plt.subplot(122)

        with torch.no_grad():

            Z = torch.zeros(100, 8)
            Z[:, model.basis.order[0]] = torch.linspace(-2, 2.2)
            f1 = model.surface(Z)

            Z = torch.zeros(100, 8)
            Z[:, model.basis.order[1]] = torch.linspace(-0.8, 1.2)
            f2 = model.surface(Z)

        m1 = df["ec50"].mean()
        s1 = df["ec50"].std()
        m2 = df["ginf"].mean()
        s2 = df["ginf"].std()

        plt.scatter(10 ** (0.2378 * s1 + m1), 10 ** (0.4157 * s2 + m2), c="k", zorder=100)

        def plotParamFill(mu, std, m1, s1, m2, s2, color):
            mu1 = mu[:, 0]
            std1 = std[:, 0]
            mu2 = mu[:, 1]
            std2 = std[:, 1]
            xf = np.concatenate((mu1 - 2 * std1, (mu1 + 2 * std1)[::-1]))
            yf = np.concatenate((mu2 - 2 * std2, (mu2 + 2 * std2)[::-1]))

            plt.fill(10 ** (xf * s1 + m1), 10 ** (yf * s2 + m2), alpha=0.6, color=color)

        plt.plot(
            10 ** (f1.mean[:, 0] * s1 + m1),
            10 ** (f1.mean[:, 1] * s2 + m2),
            label=r"$\mathbf{z}_1$",
            c="C4",
        )
        plotParamFill(f1.mean.numpy(), f1.variance.numpy() ** 0.5, m1, s1, m2, s2, "C4")

        plt.plot(
            10 ** (f2.mean[:, 0] * s1 + m1),
            10 ** (f2.mean[:, 1] * s2 + m2),
            label=r"$\mathbf{z}_2$",
            c="C8",
        )
        plotParamFill(f2.mean.numpy(), f2.variance.numpy() ** 0.5, m1, s1, m2, s2, "C8")

        plt.semilogx()
        plt.xticks([10, 100, 1000])
        plt.semilogy()
        plt.xlabel(r"$\mathrm{EC}_{50}$")

        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )

        plt.tight_layout()

        plt.savefig(output[0], bbox_inches="tight")

rule covid_anglehist:
    """
    Highlighted angle historgram for sars-cov2 data
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/anglehist-highlight.png"
    run:
        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        X = ds[: len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)

        W = model.basis.W_mu.detach().numpy()[:, model.basis.order]

        w1 = np.mean(W, axis=0)
        n1 = w1 / np.linalg.norm(w1)

        theta = np.arctan2(W[:, 1], W[:, 0])
        H, edges = np.histogram(theta, bins=100, density=True)

        plt.figure(dpi=100, figsize=(3, 3))
        ax = plt.subplot(111, polar="true",)

        bars = ax.bar(
            edges[1:], H, width=edges[1:] - edges[:-1], bottom=H.max() * 0.5, zorder=100
        )

        (ind,) = np.where(H == H.max())
        angle = edges[ind + 1]
        ax.plot(
            [angle] * 2,
            [0, H.max() * 1.5],
            c="C2",
            zorder=101,
            label="Stability axis",
            lw=2,
        )

        ax.set_yticklabels([])

        n1 = np.array([np.cos(angle)[0], np.sin(angle)[0]])

        w2 = -np.ones(2)
        n2 = w2 / np.linalg.norm(w2)
        angle2 = np.arctan2(n2[1], n2[0])

        ax.plot(
            [angle2] * 2,
            [0, H.max() * 1.5],
            c="C4",
            zorder=101,
            label="Binding axis",
            lw=2,
        )
        plt.legend()
        plt.savefig(output[0], bbox_inches="tight")

rule covid_axes_surface:
    """
    Parametric plot of binding/expression versus different axes
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/axes-surface.png"
    run:
        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        z1 = torch.zeros(100, 8)
        z2 = torch.zeros(100, 8)

        X = ds[: len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)

        W = model.basis.W_mu.detach().numpy()[:, model.basis.order]

        w1 = np.mean(W, axis=0)
        n1 = w1 / np.linalg.norm(w1)
        w2 = -np.ones(2)
        n2 = w2 / np.linalg.norm(w2)
        angle2 = np.arctan2(n2[1], n2[0])

        theta = np.arctan2(W[:, 1], W[:, 0])
        H, edges = np.histogram(theta, bins=100, density=True)

        plt.figure(dpi=100, figsize=(3, 3))
        ax = plt.subplot(111, polar="true",)

        bars = ax.bar(
            edges[1:], H, width=edges[1:] - edges[:-1], bottom=H.max() * 0.5, zorder=100
        )

        (ind,) = np.where(H == H.max())
        angle = edges[ind + 1]
        ax.plot(
            [angle] * 2,
            [0, H.max() * 1.5],
            c="C2",
            zorder=101,
            label="Stability axis",
            lw=2,
        )

        ax.set_yticklabels([])

        n1 = np.array([np.cos(angle)[0], np.sin(angle)[0]])
        span = torch.linspace(-0.2, 5.8)
        d0, d1 = model.basis.order[:2]
        z1[:, d0] = span * n1[0]
        z1[:, d1] = span * n1[1]
        z2[:, d0] = span * n2[0]
        z2[:, d1] = span * n2[1]

        with torch.no_grad():
            f1 = model.surface(z1)
            f2 = model.surface(z2)

            mu1 = f1.mean
            mu2 = f2.mean
            lw1, hi1 = f1.confidence_region()
            lw2, hi2 = f2.confidence_region()

        plt.figure(figsize=(3, 3), dpi=200)

        plt.subplot(211)
        mu = df["func_score_bind"].mean()
        std = df["func_score_bind"].std()
        plt.plot(span, f1.mean[:, 1] * std + mu, c="C2")
        plt.fill_between(
            span, lw1[:, 1] * std + mu, hi1[:, 1] * std + mu, color="C2", alpha=0.8
        )

        plt.plot(span, f2.mean[:, 1] * std + mu, c="C4")
        plt.fill_between(
            span, lw2[:, 1] * std + mu, hi2[:, 1] * std + mu, color="C4", alpha=0.8
        )
        plt.ylabel("$\\log K_D$")
        plt.xticks([])
        plt.axvline(0, c="k", ls="--")

        plt.subplot(212)
        mu = df["func_score_exp"].mean()
        std = df["func_score_exp"].std()
        plt.plot(span, f1.mean[:, 0] * std + mu, c="C2")
        plt.fill_between(
            span, lw1[:, 0] * std + mu, hi1[:, 0] * std + mu, color="C2", alpha=0.8
        )

        plt.plot(span, f2.mean[:, 0] * std + mu, c="C4")
        plt.fill_between(
            span, lw2[:, 0] * std + mu, hi2[:, 0] * std + mu, color="C4", alpha=0.8
        )
        plt.xlabel("projection")
        plt.ylabel(r"$\Delta\log$MFI")
        plt.axvline(0, c="k", ls="--")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

rule covid_parametric:
    """
    Parametric plot of binding/expression versus different axes
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/parametric.png"
    run:
        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        z1 = torch.zeros(100, 8)
        z2 = torch.zeros(100, 8)

        X = ds[: len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)

        W = model.basis.W_mu.detach().numpy()[:, model.basis.order]

        w1 = np.mean(W, axis=0)
        n1 = w1 / np.linalg.norm(w1)
        w2 = -np.ones(2)
        n2 = w2 / np.linalg.norm(w2)
        angle2 = np.arctan2(n2[1], n2[0])

        theta = np.arctan2(W[:, 1], W[:, 0])
        H, edges = np.histogram(theta, bins=100, density=True)

        plt.figure(dpi=100, figsize=(3, 3))
        ax = plt.subplot(111, polar="true",)

        bars = ax.bar(
            edges[1:], H, width=edges[1:] - edges[:-1], bottom=H.max() * 0.5, zorder=100
        )

        (ind,) = np.where(H == H.max())
        angle = edges[ind + 1]
        ax.plot(
            [angle] * 2,
            [0, H.max() * 1.5],
            c="C2",
            zorder=101,
            label="Stability axis",
            lw=2,
        )

        ax.set_yticklabels([])

        n1 = np.array([np.cos(angle)[0], np.sin(angle)[0]])
        span = torch.linspace(-0.2, 5.8)
        d0, d1 = model.basis.order[:2]
        z1[:, d0] = span * n1[0]
        z1[:, d1] = span * n1[1]
        z2[:, d0] = span * n2[0]
        z2[:, d1] = span * n2[1]

        with torch.no_grad():
            f1 = model.surface(z1)
            f2 = model.surface(z2)

            mu1 = f1.mean
            mu2 = f2.mean
            lw1, hi1 = f1.confidence_region()
            lw2, hi2 = f2.confidence_region()

        def plotParamFill(mu, std, m1, s1, m2, s2, color):
            mu1 = mu[:, 0]
            std1 = std[:, 0]
            mu2 = mu[:, 1]
            std2 = std[:, 1]
            xf = np.concatenate((mu1 + 2 * std1, (mu1 - 2 * std1)[::-1]))
            yf = np.concatenate((mu2 - 2 * std2, (mu2 + 2 * std2)[::-1]))

            plt.fill((xf * s1 + m1), (yf * s2 + m2), alpha=0.6, color=color)

        plt.figure(figsize=(5, 3), dpi=200)

        m1 = df["func_score_exp"].mean()
        s1 = df["func_score_exp"].std()
        m2 = df["func_score_bind"].mean()
        s2 = df["func_score_bind"].std()
        plotParamFill(f1.mean.numpy(), f1.variance.numpy() ** 0.5, m1, s1, m2, s2, "C2")
        plotParamFill(f2.mean.numpy(), f2.variance.numpy() ** 0.5, m1, s1, m2, s2, "C4")

        plt.plot(
            f1.mean[:, 0] * s1 + m1,
            f1.mean[:, 1] * s2 + m2,
            label=r"Stability axis",
            c="C2",
        )

        plt.plot(
            f2.mean[:, 0] * s1 + m1,
            f2.mean[:, 1] * s2 + m2,
            label=r"Binding axis",
            c="C4",
        )

        plt.ylabel("$\\log K_D$")
        plt.xlabel(r"$\Delta\log$MFI")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=True)


rule covid_variants:
    """
    Parametric plot of binding/expression versus different axes
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/variants.png",
        "figures/covid-joint/variants_project.png"
    run:
        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        subs = ["N171Y", "L122R", "E154K"]
        labels = ["N501Y", "L452R", "E484K"]

        W = model.basis.W_mu.detach().numpy()
        N = W / np.linalg.norm(W, axis=1)[:, None]

        ind = np.array([ds.tokenizer.tokens.index(s) for s in subs])
        plt.plot(W[ind, :][:, model.basis.order].T)
        plt.plot(np.mean(W[ind, :][:, model.basis.order], axis=0).T, c="k")

        np.mean(W[ind, :], axis=0)

        X, y, n = ds[: len(ds)]
        sel = (X.sum(axis=1) <= 1) & (n[:, 0] < 1)

        with torch.no_grad():
            Z = model.basis(X)
            f = model(X[sel, :])

            Z = Z[:, model.basis.order]


        d0 = model.basis.order[0]
        d1 = model.basis.order[1]

        span = torch.linspace(-0.4, 5)
        basis = torch.from_numpy(W[ind[0], :])
        basis = basis / basis.norm()

        z, fmu, fvar, Z1, Z2, y, _Z = util.buildLandscape(
            model,
            ds,
            p=1,
            mu=df["func_score_bind"].mean(),
            std=df["func_score_bind"].std(),
        )

        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            levels=50,
            mask=False,
            image=False,
            plotOrigin=False,
            fig=fig,
            ax=ax,
        )

        _, _, _, im = plt.hist2d(
            Z[:, 0].numpy(),
            Z[:, 1].numpy(),
            bins=(np.linspace(-0.4, 2.2), np.linspace(-0.4, 2.2)),
            norm=mpl.colors.LogNorm(),
            cmap="Oranges",
        )

        plt.xlim(-0.4, 2.2)
        plt.ylim(-0.2, 2.2)

        for i in range(3):
            plt.arrow(
                0,
                0,
                W[ind[i], d0],
                W[ind[i], d1],
                fc=f"C{i+4}",
                label=labels[i],
                width=0.025,
                zorder=99,
            )

        ax.set_facecolor("Grey")
        ax.legend(ncol=2)

        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")

        """Second figure
        """
        N = W / np.linalg.norm(W, axis=1)[:, None]

        proj = np.dot(W, basis.numpy()) / basis.norm()
        scores = torch.from_numpy(np.dot(N, basis.numpy() / basis.norm()))

        nrm = np.linalg.norm(W, axis=1)

        span = torch.linspace(-0.3, 2.1)
        Z = basis.repeat(100, 1) * span.reshape(100, 1)

        mu = df["func_score_bind"].mean()
        std = df["func_score_bind"].std()
        with torch.no_grad():
            f = model.surface(Z)
            lower, upper = f.confidence_region()

        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        plt.plot(span, f.mean[:, 1] * std + mu, zorder=0, color="k")
        plt.fill_between(
            span,
            lower[:, 1] * std + mu,
            upper[:, 1] * std + mu,
            alpha=0.6,
            zorder=0,
            color="k",
        )

        # projection scores
        ranks = torch.argsort(scores).numpy()[::-1][:10]

        cm = plt.get_cmap("hsv")
        colors = cm(np.linspace(0, 1, len(ranks)))
        for i, r in enumerate(ranks):

            tok = ds.tokenizer.tokens[r]
            lab = "{}{}{}".format(tok[0], int(tok[1:-1]) + 330, tok[-1])
            if (df.aa_substitutions == tok).any():
                tmp = df[df.aa_substitutions == tok]
                mmu = tmp.func_score_bind.mean()
                sem = (tmp.func_score_var_bind.mean() / tmp.shape[0]) ** 0.5
                plt.errorbar(proj[r], mmu, sem, c=colors[i], label=lab, fmt="o")

        plt.axvline(0, c="r", ls="--")
        plt.xlabel("projection")
        plt.ylabel("predicted $K_d$")

        fig.legend(
            loc="lower left",
            ncol=5,
            bbox_to_anchor=(0.0, 0.9),
            labelspacing=0.1,
            handletextpad=-0.5,
            columnspacing=0.3,
        )

        plt.savefig(output[1], bbox_inches="tight", transparent=True)
