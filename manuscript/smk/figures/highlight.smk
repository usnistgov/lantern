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
    group: "figure"
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
            fig_kwargs=dict(dpi=300, figsize=(3, 2)),
            cbar_kwargs = dict(aspect=5, shrink=0.6),
            contour_kwargs=dict(linewidths=3)
        )
        fig.axes[-1].set_title("avGFP\nBrightness", ha="left", loc="left")

        plt.savefig(output[0], bbox_inches="tight", transparent=True)

rule gfp_surface_z2:
    """
    Slice of z2 for avgfp
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt",
    output:
        "figures/gfp-brightness/brightness/surface-z2.png"
    group: "figure"
    run:
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()

        raw = "medianBrightness"

        z1pos = 1.8
        X, y = ds[: len(ds)]
        with torch.no_grad():
            Z = model.basis(X)

            zpred = torch.zeros(100, 8)
            zpred[:, model.basis.order[1]] = torch.linspace(
                Z[:, model.basis.order[1]].min(), Z[:, model.basis.order[1]].max()
            )
            zpred[:, model.basis.order[0]] = z1pos

            fpred = model.surface(zpred)
            lo, hi = fpred.confidence_region()

        ys = (Z[:, model.basis.order[0]] > z1pos - 0.3) & (Z[:, model.basis.order[0]] < z1pos + 0.3)

        plt.hist2d(
            Z[ys, model.basis.order[1]].numpy(),
            y[ys, 0].numpy() * df[raw].std() + df[raw].mean(),
            bins=30,
            norm=mpl.colors.LogNorm(),
        )
        plt.plot(
            zpred[:, model.basis.order[1]].numpy(),
            fpred.mean * df[raw].std() + df[raw].mean(),
            lw=3,
            c="C1",
            label="f(z)",
        )
        plt.fill_between(
            zpred[:, model.basis.order[1]].numpy(),
            lo.numpy() * df[raw].std() + df[raw].mean(),
            hi.numpy() * df[raw].std() + df[raw].mean(),
            alpha=0.6,
            color="C1",
        )
        plt.xlabel("$z_2$")
        plt.ylabel("avGFP Brightness")
        plt.legend()
        plt.ylim(2.4, 4.1)
        plt.savefig(output[0], bbox_inches="tight", transparent=True)

rule gfp_surface_bfp1:
    """
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/brightness/surface-bfp1.png"
    group: "figure"
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
    group: "figure"
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

rule gfp_surface_bfp_all:
    """
    """

    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/brightness/surface-bfp-all.png"
    group: "figure"
    run:
        import matplotlib.patches as mpatches

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

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=False,
            image=False,
            mask=False,
            fig = fig,
            ax = ax,
            plotOrigin=False,
            levels=20,
            cbar_kwargs = dict(aspect=5, shrink=0.6),
        )
        fig.axes[-1].set_title("avGFP\nBrightness", ha="left", loc="left")

        X, y = ds[: len(ds)][:2]

        # get variant info
        w_mu = model.basis.W_mu.detach()[:, model.basis.order].numpy()

        wt = np.zeros(w_mu.shape[1])


        # foundational
        subs = "SY64H".split(":")
        labels = ["Y66H"]
        ind = [ds.tokenizer.tokens.index(s) for s in subs]

        # from wild-type
        for i in ind:
            _z = w_mu[i, :]

            arrow = mpatches.FancyArrowPatch(
                (0, 0),
                (_z[0], _z[1]),
                mutation_scale=20,
                color="black",
                label="+{}".format(labels[ind.index(i)]),
                zorder=101,
                shrinkA=0,
                shrinkB=0,
            )
            ax.add_patch(arrow)

            # plt.arrow(
            #     wt[0],
            #     wt[1],
            #     _z[0],
            #     _z[1],
            #     color="black",
            #     label="+{}".format(labels[ind.index(i)]),
            #     length_includes_head=True,
            #     width=0.02,
            #     zorder=100,
            # )


        # substitutions
        subs = ["SY143F", "SS63T", "SH229L",'SY37N', 'SN103T', 'SY143F', 'SI169V', 'SA204V', 'SN196S']
        labels = ["Y145F", "S65T", "H231L",'Y39N', 'N105T', 'Y145F', 'I171V', 'A206V', 'N198S', ]
        ind = [ds.tokenizer.tokens.index(s) for s in subs]

        # from wild-type
        for i in ind:
            _z = w_mu[i, :]

            arrow = mpatches.FancyArrowPatch(
                (0, 0),
                (_z[0], _z[1]),
                mutation_scale=10,
                color="C{}".format(ind.index(i)),
                label="+{}".format(labels[ind.index(i)]),
                zorder=100,
                shrinkA=1,
                shrinkB=0,
            )
            ax.add_patch(arrow)

            # plt.arrow(
            #     wt[0],
            #     wt[1],
            #     _z[0],
            #     _z[1],
            #     color="C{}".format(ind.index(i)),
            #     label="+{}".format(labels[ind.index(i)]),
            #     length_includes_head=True,
            #     width=0.01,
            #     zorder=100,
            # )

        # plt.scatter(0, 0, c="r", s=10, zorder=200)

        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=5,
            # mode="expand",
            borderaxespad=0.02,
            labelspacing=0.1,
            handletextpad=0.2,
            columnspacing=0.3,
            handlelength=0.6,
        )

        # fig.legend(
        #     ncol=1,
        #     bbox_to_anchor=(1.01, 0.9),
        #     loc="upper left",
        #     borderaxespad=0.02,
        # )
        #ax.set_aspect(1.0)

        plt.savefig(output[0], bbox_inches="tight")


rule gfp_effects_cutoff:
    input:
        "data/processed/gfp.csv",
        "data/processed/gfp-brightness.pkl",
        "experiments/gfp-brightness/lantern/full/model.pt"
    output:
        "figures/gfp-brightness/effects-cutoff.png"
    group: "figure"
    run:
        from scipy.stats import norm
        
        df, ds, model = util.load_run("gfp", "brightness", "lantern", "full", 8)
        model.eval()

        qW = norm(
            model.basis.W_mu[:, model.basis.order[0]].detach().numpy(),
            model.basis.W_log_sigma[:, model.basis.order[0]].detach().exp().numpy(),
        )

        lo = qW.ppf(0.025)
        hi = qW.ppf(0.975)
        mu = qW.mean()

        # ind = [68, 595, 1108, 1845, 1005, 786, 219, 396]
        ind = [68, 595, 1108, 219]

        _, ax = plt.subplots(figsize=(4, 2))

        delta = 1.0
        offset = delta/2
        for ii, i in enumerate(ind):
            plt.plot([lo[i], hi[i]], [offset]*2, color=f"C{ii}")
            plt.scatter([mu[i]], [offset], color=f"C{ii}", label=ds.tokenizer.tokens[i][1:])

            offset += delta

        ax.spines.bottom.set_position("zero")
        ax.spines.top.set_color("none")
        ax.spines.left.set_position('zero')
        ax.spines.right.set_color('none')
        ax.xaxis.set_ticks_position("bottom")

        ax.set_xlabel("$z_1$")
        ax.xaxis.set_label_coords(-0.05, -0.025)

        # ax.yaxis.set_ticks_position("left")
        plt.yticks([])
        # plt.xticks(range(len(features)), features, rotation=45, horizontalalignment="right")

        # from here: https://www.py4u.net/discuss/139791
        # plt.setp(ax.get_xticklabels(), transform=ax.get_xaxis_transform())

        plt.grid(True, axis="x", ls="--")
        fig = plt.gcf()
        fig.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
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
    group: "figure"
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
            label=r"$K_I$",
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
            label=r"$K_A$",
        )

        plt.scatter(allostery.ec50(), allostery.ginf(), c="k", zorder=100)

        plt.semilogx()
        plt.semilogy()

        plt.xlabel(r"$\mathrm{EC}_{50}$")
        plt.ylabel(r"$\mathrm{G}_{\infty} / \mathrm{G}_{\infty}^{max}$")

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

        plt.scatter(
            10 ** (0.2378 * s1 + m1),
            # 10 ** (0.4157 * s2 + m2) / (10 ** (0.4157 * s2 + m2)).max(),
            10 ** (0.4157 * s2 + m2) / 10**df["ginf"].max(),
            c="k",
            zorder=100,
        )

        def plotParamFill(mu, std, m1, s1, m2, s2, color):
            mu1 = mu[:, 0]
            std1 = std[:, 0]
            mu2 = mu[:, 1]
            std2 = std[:, 1]
            xf = np.concatenate((mu1 - 2 * std1, (mu1 + 2 * std1)[::-1]))
            yf = np.concatenate((mu2 - 2 * std2, (mu2 + 2 * std2)[::-1]))

            plt.fill(
                10 ** (xf * s1 + m1),
                # 10 ** (yf * s2 + m2) / 10 ** (yf * s2 + m2).max(),
                10 ** (yf * s2 + m2) / 10 ** df["ginf"].max(),
                alpha=0.6,
                color=color,
            )

        plt.plot(
            10 ** (f1.mean[:, 0] * s1 + m1),
            # 10 ** (f1.mean[:, 1] * s2 + m2) / (10 ** (f1.mean[:, 1] * s2 + m2)).max(),
            10 ** (f1.mean[:, 1] * s2 + m2) / (10 ** df["ginf"]).max(),
            label=r"$\mathbf{z}_1$",
            c="C4",
        )
        plotParamFill(f1.mean.numpy(), f1.variance.numpy() ** 0.5, m1, s1, m2, s2, "C4")

        plt.plot(
            10 ** (f2.mean[:, 0] * s1 + m1),
            # 10 ** (f2.mean[:, 1] * s2 + m2) / (10 ** (f2.mean[:, 1] * s2 + m2)).max(),
            10 ** (f2.mean[:, 1] * s2 + m2) / 10**df["ginf"].max(),
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

rule laci_surface_z1:
    """
    Slice of z1 for laci
    """

    input:
        "data/processed/laci.csv",
        "data/processed/laci-joint.pkl",
        "experiments/laci-joint/lantern/full/model.pt",
    output:
        "figures/laci-joint/ec50/surface-z1.png"
    group: "figure"
    run:
        df, ds, model = util.load_run("laci", "joint", "lantern", "full", 8)
        model.eval()

        raw = "ec50-norm"

        z2pos = 0
        X, y = ds[: len(ds)][:2]
        with torch.no_grad():
            Z = model.basis(X)

            zpred = torch.zeros(100, 8)
            zpred[:, model.basis.order[0]] = torch.linspace(
                Z[:, model.basis.order[0]].min(), Z[:, model.basis.order[0]].max()
            )
            zpred[:, model.basis.order[1]] = z2pos

            fpred = model.surface(zpred)
            lo, hi = fpred.confidence_region()

        ys = (Z[:, model.basis.order[1]] > z2pos - 0.3) & (Z[:, model.basis.order[1]] < z2pos + 0.3)

        plt.hist2d(
            Z[ys, model.basis.order[0]].numpy(),
            y[ys, 0].numpy() * df[raw].std() + df[raw].mean(),
            bins=30,
            norm=mpl.colors.LogNorm(),
        )
        plt.plot(
            zpred[:, model.basis.order[0]].numpy(),
            fpred.mean[:, 0] * df[raw].std() + df[raw].mean(),
            lw=3,
            c="C1",
            label="f(z)",
        )
        plt.fill_between(
            zpred[:, model.basis.order[0]].numpy(),
            lo[:, 0].numpy() * df[raw].std() + df[raw].mean(),
            hi[:, 0].numpy() * df[raw].std() + df[raw].mean(),
            alpha=0.6,
            color="C1",
        )
        plt.xlabel("$z_1$")
        plt.ylabel(r"LacI $\mathrm{EC}_{50}$")
        plt.legend()
        plt.savefig(output[0], bbox_inches="tight", transparent=True)

COVID_BINDING_COLOR = "fuchsia"
COVID_STABILITY_COLOR = "limegreen"

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
    group: "figure"
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
        H, edges = np.histogram(theta, bins=50, density=True)

        fig = plt.figure(dpi=300, figsize=(2, 2))
        ax = plt.subplot(111, polar="true",)

        bars = ax.bar(
            edges[1:], H, width=edges[1:] - edges[:-1], bottom=H.max() * 0.5, zorder=100
        )

        (ind,) = np.where(H == H.max())
        angle = edges[ind + 1]
        # ax.plot(
        #     [angle] * 2,
        #     [0, H.max() * 1.5],
        #     c=COVID_STABILITY_COLOR,
        #     zorder=101,
        #     label="Stability\naxis",
        #     lw=2,
        # )
        ax.arrow(
            angle,
            0,
            0,
            H.max()*1.8,
            fc=COVID_STABILITY_COLOR,
            zorder=90,
            label="Stability axis",
            width=0.05,
        )

        ax.set_yticklabels([])

        n1 = np.array([np.cos(angle)[0], np.sin(angle)[0]])

        w2 = -np.ones(2)
        n2 = w2 / np.linalg.norm(w2)
        angle2 = np.arctan2(n2[1], n2[0])

        # ax.plot(
        #     [angle2] * 2,
        #     [0, H.max() * 1.5],
        #     c=COVID_BINDING_COLOR,
        #     zorder=101,
        #     label="Binding\naxis",
        #     lw=2,
        # )
        ax.arrow(
            angle2,
            0,
            0,
            H.max()*1.8,
            fc=COVID_BINDING_COLOR,
            zorder=90,
            label="Binding axis",
            width=0.05,
        )

        # legend above
        # plt.legend(
        #     bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        #     loc="lower left",
        #     ncol=2,
        #     mode="expand",
        #     borderaxespad=0.0,
        # )

        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        # plt.legend(
        #     bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        #     loc="lower left",
        #     ncol=1,
        #     mode="expand",
        #     borderaxespad=0.0,
        # )
        # plt.legend(
        #     bbox_to_anchor=(1.05, 1.04),
        #     loc="upper right",
        #     borderaxespad=0.0,
        #     ncol=1,
        #     labelspacing=0.1,
        #     handletextpad=0.2,
        #     columnspacing=0.3,
        #     handlelength=0.6
        # )

        plt.savefig(output[0], bbox_inches="tight", transparent=True)

rule covid_axes_parametric:
    """
    Parametric plot of binding/expression versus different axes
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/axes-surface.png"
    group: "figure"
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

        plt.figure(figsize=(6, 2), dpi=200)

        plt.subplot(121)
        mu = df["func_score_bind"].mean()
        std = df["func_score_bind"].std()
        plt.plot(span, f1.mean[:, 1] * std + mu, c=COVID_STABILITY_COLOR)
        plt.fill_between(
            span, lw1[:, 1] * std + mu, hi1[:, 1] * std + mu, color=COVID_STABILITY_COLOR, alpha=0.4
        )

        plt.plot(span, f2.mean[:, 1] * std + mu, c=COVID_BINDING_COLOR)
        plt.fill_between(
            span, lw2[:, 1] * std + mu, hi2[:, 1] * std + mu, color=COVID_BINDING_COLOR, alpha=0.4
        )
        #plt.xticks([])
        plt.xlabel("axis coordinate")
        plt.axvline(0, c="k", ls="--")
        plt.ylabel("RBD-ACE2 $\log_{10} K_d$")

        plt.subplot(122)
        mu = df["func_score_exp"].mean()
        std = df["func_score_exp"].std()
        plt.plot(span, f1.mean[:, 0] * std + mu, c=COVID_STABILITY_COLOR)
        plt.fill_between(
            span, lw1[:, 0] * std + mu, hi1[:, 0] * std + mu, color=COVID_STABILITY_COLOR, alpha=0.4
        )

        plt.plot(span, f2.mean[:, 0] * std + mu, c=COVID_BINDING_COLOR)
        plt.fill_between(
            span, lw2[:, 0] * std + mu, hi2[:, 0] * std + mu, color=COVID_BINDING_COLOR, alpha=0.4
        )
        plt.xlabel("axis coordinate")
        plt.ylabel("RBD $\Delta\log$MFI")
        plt.axvline(0, c="k", ls="--")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=True)

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
    group: "figure"
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

        plt.figure(figsize=(3, 2), dpi=200)

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

        plt.ylabel("$\\log_{10} K_D$")
        plt.xlabel(r"$\Delta\log$MFI")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=True)


rule covid_variants:
    """
    Covid variants of interest
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/variants.png",
        "figures/covid-joint/variants_project.png"
    group: "figure"
    run:
        from cycler import cycler

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
            # f = model(X[sel, :])
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

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            levels=100,
            mask=False,
            image=False,
            plotOrigin=False,
            fig=fig,
            ax=ax,
            colorbar=True,
            # contour_label="RBD-ACE2 $\log K_d$",
            contour_kwargs=dict(linewidths=3),
            cbar_kwargs=dict(aspect=5, shrink=0.8)
        )

        # _, _, _, im = plt.hist2d(
        #     Z[:, 0].numpy(),
        #     Z[:, 1].numpy(),
        #     bins=(np.linspace(-0.4, 2.2), np.linspace(-0.4, 2.2)),
        #     norm=mpl.colors.LogNorm(),
        #     cmap="Oranges",
        # )

        plt.xlim(-0.4, 2.2)
        plt.ylim(-0.2, 2.2)

        colors = ["coral", "mediumorchid", "palegreen"]

        for i in range(3):
            plt.arrow(
                0,
                0,
                W[ind[i], d0],
                W[ind[i], d1],
                # fc=f"C{i+4}",
                fc=colors[i],
                label=labels[i],
                width=0.05,
                zorder=99,
            )

        # ax.set_facecolor("Grey")
        ax.legend(ncol=1)

        # add title to cbar axis
        fig.axes[-1].set_title("RBD-ACE2\n$\log_{10} K_d$", y=1.04, ha="left", loc="left")

        # fig.colorbar(im, ax=ax, location="bottom", pad=0.2)

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=False)

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

        fig, ax = plt.subplots(figsize=(2, 2), dpi=200)
        plt.plot(span, f.mean[:, 1] * std + mu, zorder=0, color="k")
        plt.fill_between(
            span,
            lower[:, 1] * std + mu,
            upper[:, 1] * std + mu,
            alpha=0.2,
            zorder=0,
            color="k",
        )

        # projection scores
        ranks = torch.argsort(scores).numpy()[::-1][:10]

        # remove N501Y
        ranks = [r for r in ranks if r != ind[0]]

        # remove those without single-mutant data

        # cm = plt.get_cmap("hsv")
        # colors = cm(np.linspace(0, 1, len(ranks)))
        # color.cycle_cmap(8, cmap='pastel', ax=ax)
        colors = plt.get_cmap("Set1")(np.linspace(0, 1, 9))
        cind = 0
        # ax.set_prop_cycle(cycler(color=color))
        anchor = None
        shift = 0.05
        for i, r in enumerate(ranks):

            tok = ds.tokenizer.tokens[r]
            lab = "{}{}{}".format(tok[0], int(tok[1:-1]) + 330, tok[-1])
            if (df.aa_substitutions == tok).any():
                tmp = df[df.aa_substitutions == tok]
                mmu = tmp.func_score_bind.mean()
                sem = (tmp.func_score_var_bind.mean() / tmp.shape[0]) ** 0.5

                offset = 0
                if lab == "Q498W":
                    offset = -shift
                    anchor = (proj[r], mmu)
                elif lab == "Y453F":
                    offset = shift

                plt.errorbar(
                    proj[r] + offset, mmu, sem, c=colors[cind], label=lab, fmt="o"
                )
                # plt.errorbar(proj[r], mmu, sem, c=f"C{i}", label=lab, fmt="o")

                cind += 1

        # plt.plot(
        #     [anchor[0] - shift, anchor[0] + shift],
        #     [anchor[1]] * 2,
        #     # ls="--",
        #     color="k",
        #     # zorder=0,
        # )
        # plt.scatter(*anchor, c="k", zorder=0, s=20, marker="d")

        # plot N501Y
        tmp = df[df.aa_substitutions == subs[0]]
        mmu = tmp.func_score_bind.mean()
        sem = (tmp.func_score_var_bind.mean() / tmp.shape[0]) ** 0.5
        plt.errorbar(proj[ind[0]], mmu, sem, c="k", label="N501Y", fmt="o")
        

        plt.axvline(0, c="r", ls="--")
        plt.xlabel("projection")
        plt.ylabel("RBD-ACE2 $\log K_d$")

        # upper legend
        # fig.legend(
        #     loc="lower left",
        #     ncol=5,
        #     bbox_to_anchor=(0.0, 0.9),
        #     labelspacing=0.1,
        #     handletextpad=-0.5,
        #     columnspacing=0.3,
        # )

        # lower legend
        # fig.legend(
        #     loc="upper left",
        #     ncol=4,
        #     bbox_to_anchor=(0.0, 0.1),
        #     labelspacing=0.1,
        #     handletextpad=-0.5,
        #     columnspacing=0.3,
        # )
        
        # rhs legend
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            ncol=1,
            labelspacing=0.1,
            handletextpad=-0.5,
            columnspacing=0.3,
        )

        plt.savefig(output[1], bbox_inches="tight", transparent=True)

rule covid_axes_surface:
    """
    Covid axes surfaces
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/axes-binding.png",
        "figures/covid-joint/axes-expression.png",
        # "figures/covid-joint/axes-expression-with-hist.png"
    group: "figure"
    run:

        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        X = ds[:len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        W = model.basis.W_mu[:, model.basis.order].detach().cpu().numpy()

        w1 = np.mean(W, axis=0)
        n1 = w1 / np.linalg.norm(w1)
        w2 = -np.ones(2)
        n2 = w2 / np.linalg.norm(w2)

        """Binding surface
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        d0 = model.basis.order[0]
        d1 = model.basis.order[1]

        span = torch.linspace(-10, 10, 100)

        z, fmu, fvar, Z1, Z2, _y, _Z = util.buildLandscape(
            model,
            ds,
            p=1,
            mu=df["func_score_bind"].mean(),
            std=df["func_score_bind"].std(),
        )


        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

        lw, hi = (-3.1, 1.8)

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            levels=30,
            mask=False,
            image=False,
            plotOrigin=False,
            fig=fig,
            ax=ax,
            cbar_kwargs=dict(aspect=5, shrink=0.8)
            # contour_kwargs=dict(linewidths=3)
        )

        # _, _, _, im = plt.hist2d(
        #     Z[:, 0].numpy(),
        #     Z[:, 1].numpy(),
        #     bins=(np.linspace(lw, hi), np.linspace(lw, hi)),
        #     norm=mpl.colors.LogNorm(),
        #     cmap="Reds",
        # )

        plt.xlim(lw, hi)
        plt.ylim(lw, hi)

        plt.arrow(
            0,
            0,
            n1[0].item(),
            n1[1].item(),
            width=0.10,
            zorder=100,
            fc=COVID_STABILITY_COLOR,
            label="Stability axis",
        )

        plt.plot(n2[0].item() * span, n2[1].item() * span, ls="--", c="k")
        plt.arrow(
            0,
            0,
            n2[0].item(),
            n2[1].item(),
            width=0.10,
            zorder=100,
            fc=COVID_BINDING_COLOR,
            label="Binding axis",
        )

        # ax.set_facecolor("Grey")
        # ax.legend()

        # fig.colorbar(im, ax=ax)

        # colorbar title
        fig.axes[-1].set_title("RBD-ACE2\n$\log_{10} K_d$", y=1.04, loc="left", ha="left")

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=False)

        """Expression surface
        """

        z, fmu, fvar, Z1, Z2, _y, _Z = util.buildLandscape(
            model,
            ds,
            p=0,
            mu=df["func_score_exp"].mean(),
            std=df["func_score_exp"].std(),
        )


        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

        lw, hi = (-3.1, 1.8)

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            levels=30,
            mask=False,
            image=False,
            plotOrigin=False,
            fig=fig,
            ax=ax,
            cbar_kwargs=dict(aspect=5, shrink=0.8)
            # contour_kwargs=dict(linewidths=3)
        )

        # _, _, _, im = plt.hist2d(
        #     Z[:, 0].numpy(),
        #     Z[:, 1].numpy(),
        #     bins=(np.linspace(lw, hi), np.linspace(lw, hi)),
        #     norm=mpl.colors.LogNorm(),
        #     cmap="Reds",
        # )


        plt.xlim(lw, hi)
        plt.ylim(lw, hi)

        plt.arrow(
            0,
            0,
            n1[0].item(),
            n1[1].item(),
            width=0.10,
            zorder=100,
            fc=COVID_STABILITY_COLOR,
            label="Stability axis",
        )

        plt.plot(n2[0].item() * span, n2[1].item() * span, ls="--", c="k")
        plt.arrow(
            0,
            0,
            n2[0].item(),
            n2[1].item(),
            width=0.10,
            zorder=100,
            fc=COVID_BINDING_COLOR,
            label="Binding axis",
        )

        # ax.set_facecolor("Grey")

        # colorbar title
        fig.axes[-1].set_title("RBD\n$\Delta \log$MFI", y=1.04, loc="left", ha="left")

        plt.tight_layout()
        plt.savefig(output[1], bbox_inches="tight", transparent=False)

        # fig.colorbar(im, ax=ax, location="bottom", pad=0.2)
        # plt.savefig(output[2], bbox_inches="tight", transparent=False)

rule covid_bind_surface_zk:
    """
    Covid axes surfaces slice
    """

    input:
        "data/processed/covid.csv",
        "data/processed/covid-joint.pkl",
        "experiments/covid-joint/lantern/full/model.pt"
    output:
        "figures/covid-joint/surface-binding-z{k}.png",
    group: "figure"
    run:

        df, ds, model = util.load_run("covid", "joint", "lantern", "full", 8)
        model.eval()

        K = int(wildcards.k) - 1

        X = ds[:len(ds)][0]
        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        Zpred = torch.zeros(100, 8)

        rng = Z[:, K].max() - Z[:, K].min()
        Zpred[:, model.basis.order[K]] = torch.linspace(-3, 3)

        for z1 in [0, -2, -4, -6, -8, -10]:
            Zpred[:, model.basis.order[0]] = z1

            with torch.no_grad():
                f = model.surface(Zpred)
                lo, hi = f.confidence_region()

            plt.plot(
                Zpred[:, model.basis.order[K]].numpy(),
                f.mean[:, 1].numpy(),
                label=f"$z_1$ = {z1}",
            )
            plt.fill_between(
                Zpred[:, model.basis.order[K]].numpy(),
                lo[:, 1].numpy(),
                hi[:, 1].numpy(),
                alpha=0.4,
            )

        plt.xlabel(f"$z_{K+1}$")
        plt.ylabel("SARS-CoV-2 binding (normalized)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(output[0], bbox_inches="tight", transparent=False)

rule allostery_rotate:
    input:
        "data/processed/allostery2.csv",
        "data/processed/allostery2-joint.pkl",
        "experiments/allostery2-joint/lantern/full/model.pt"
    output:
        "figures/allostery2-joint/ec50/rotation-actual.png",
        "figures/allostery2-joint/ginf/rotation-actual.png",
        "figures/allostery2-joint/g0/rotation-actual.png",
        "figures/allostery2-joint/ec50/rotation-residual.png",
        "figures/allostery2-joint/ginf/rotation-residual.png",
        "figures/allostery2-joint/g0/rotation-residual.png",
    run:

        from src.allostery import ec50, ginf, g0, delta_eps_AI_0, delta_eps_RA_0

        df, ds, model = util.load_run("allostery2", "joint", "lantern", "full", 8,)
        
        p1 = np.linspace(df.eps_AI_shift.min(), df.eps_AI_shift.max())
        p2 = np.linspace(df.eps_RA_shift.min(), df.eps_RA_shift.max())

        # build prediction surface in biophysical space
        P1, P2 = np.meshgrid(p1, p2)
        surface = np.concatenate(
            (P1.reshape((2500, 1)), P2.reshape((2500, 1))),
            axis=1
        )

        # how to go from biophysical to learned space
        with torch.no_grad():
            _X = ds[:len(ds)][0]
            X = model.basis(_X).numpy()

        y1 = df.eps_AI_shift.values
        y2 = df.eps_RA_shift.values

        y = df[
            ["eps_AI_shift", "eps_RA_shift"]
        ].values

        V, _, _, _ = np.linalg.lstsq(y, X - X.mean(axis=0), rcond=None)

        # setup prediction
        d0, d1 = model.basis.order[:2]

        Zpred = torch.zeros(2500, 8)

        # make latent values from linear relationship
        tmp = np.dot(surface, V) + X.mean(axis=0)
        # Zpred[:, d0] = torch.from_numpy(tmp[:, 0])
        # Zpred[:, d1] = torch.from_numpy(tmp[:, 1])
        Zpred[:, d0] = torch.from_numpy(tmp[:, d0])
        Zpred[:, d1] = torch.from_numpy(tmp[:, d1])

        with torch.no_grad():
            fpred = model.surface(Zpred)

        # scale to make visible
        Vn = 50*V

        for i, (fxn, label) in enumerate(
            [
                (ec50, r"$\mathrm{EC}_{50}$"),
                (ginf, r"$\mathrm{G}_{\infty}$"),
                (g0, r"$\mathrm{G}_{0}$"),
            ]
        ):

            # Rotated view

            param = np.log10(
                fxn(
                    delta_eps_AI=surface[:, 0] + delta_eps_AI_0,
                    delta_eps_RA=surface[:, 1] + delta_eps_RA_0,
                )
            ).reshape(50, 50)

            pred = fpred.mean[:, i].numpy().reshape((50, 50))

            fig = plt.figure(figsize=(3, 2.5), dpi=300)
            im = plt.imshow(
                param,
                aspect="auto",
                interpolation="lanczos",
                extent=(P1.min(), P1.max(), P2.min(), P2.max()),
                origin="lower",
                vmin=min(param.min(), pred.min()),
                vmax=max(param.max(), pred.max()),
            )
            plt.contour(
                P1,
                P2,
                pred,
                vmin=min(param.min(), pred.min()),
                vmax=max(param.max(), pred.max()),
            )

            plt.arrow(0, 0, Vn[0, d0], Vn[1, d0], width=0.2, zorder=100)
            plt.text(Vn[0, d0], 0.8 * Vn[1, d0], "$z_1$", color="k")

            plt.arrow(0, 0, Vn[0, d1], Vn[1, d1], width=0.2, zorder=100)
            plt.text(Vn[0, d1] * 1.5, 0.6 * Vn[1, d1], "$z_2$", color="k")

            plt.xlabel("$\Delta \epsilon_{AI}$")
            plt.ylabel("$\Delta \epsilon_{RA}$")
            fig.colorbar(im)
            plt.title(label)

            plt.tight_layout()
            plt.savefig(output[i], bbox_inches="tight", transparent=False)

            # Residual view
            fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
            _, _, _, im = plt.hist2d(y1, y2, bins=30, norm=mpl.colors.LogNorm(), cmap="Greens")

            resid = (pred - param) ** 2
            cim = plt.contour(
                P1,
                P2,
                resid,
                levels=10,
                cmap="plasma"
            )

            norm = mpl.colors.Normalize(vmin=resid.min(), vmax=resid.max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cim.cmap)
            sm.set_array([])
            fig.colorbar(sm, pad=0.24, aspect=5)
            fig.axes[-1].set_title("residual", ha="left", loc="left")

            plt.arrow(0, 0, Vn[0, d0], Vn[1, d0], width=0.2, zorder=100)
            plt.text(Vn[0, d0], 0.8 * Vn[1, d0], "$z_1$", color="k")

            plt.arrow(0, 0, Vn[0, d1], Vn[1, d1], width=0.2, zorder=100)
            plt.text(Vn[0, d1] * 1.5, 0.6 * Vn[1, d1], "$z_2$", color="k")

            plt.xlabel("$\Delta \epsilon_{AI}$")
            plt.ylabel("$\Delta \epsilon_{RA}$")
            fig.colorbar(im, pad=0.05, aspect=5)
            fig.axes[-1].set_title("observation\ncount", ha="left", loc="left")

            ax.set_facecolor("Grey")

            plt.savefig(output[i+3], bbox_inches="tight", transparent=False)
