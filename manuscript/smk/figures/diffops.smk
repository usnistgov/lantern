rule diffops:
    input:
        "experiments/laci-joint/lantern/full/model.pt"
    output:
        "figures/diffops-curvature.png",
        "figures/diffops-slope.png"
    group: "figure"
    run:
        df, ds, model = util.load_run("laci", "joint", "lantern", "full", 8)

        model.eval()
        X = ds[: len(ds)][0]

        with torch.no_grad():
            Z = model.basis(X)
            Z = Z[:, model.basis.order]

        from lantern.diffops import lapl
        from lantern.diffops import grad
        from lantern.diffops import metric

        zrng = torch.linspace(-4.8, 4.8)
        Zpred = torch.zeros(100, 8)
        Zpred[:, model.basis.order[0]] = zrng

        strat = model.surface.variational_strategy
        if isinstance(strat, IndependentMultitaskVariationalStrategy):
            strat = strat.base_variational_strategy
        z0 = strat.inducing_points.detach()
        if z0.ndim > 2:
            z0 = z0[0, :, :]

        with torch.no_grad():

            fpred = model.surface(Zpred)
            lower, upper = fpred.confidence_region()

            lmu, lvar = lapl.laplacian(
                model.surface, Zpred, z0, dims=model.basis.order[:2], p=0
            )
            nrm = torch.distributions.Normal(lmu, lvar.sqrt())
            additivity = metric.kernel(lmu, lvar)

            lmu = lmu.numpy()
            lvar = lvar.numpy()

        fig, ax = plt.subplots(dpi=200, figsize=(3, 2))
        plt.plot(zrng, fpred.mean[:, 0], label="$f(z)$")
        plt.fill_between(zrng, lower[:, 0], upper[:, 0], alpha=0.6)
        plt.xlabel("$z_1$")

        plt.setp([ax.get_yticklabels()], color="C0")
        ax.tick_params(axis="y", color="C0")
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="C0", linewidth=1.0)
        ax.set_ylabel("$f(\mathbf{z})$", color="C0")

        ax = plt.twinx()
        plt.plot(zrng, lmu, c="C1", label="Laplacian")
        plt.fill_between(
            zrng,
            lmu - np.sqrt(lvar) * 2,
            lmu + np.sqrt(lvar) * 2,
            alpha=0.6,
            color="C1",
        )
        plt.axhline(0, c="C1", ls="--")

        plt.setp([ax.get_yticklabels()], color="C1")
        ax.tick_params(axis="y", color="C1")
        for pos in [
            "right",
        ]:
            plt.setp(ax.spines[pos], color="C1", linewidth=1.0)
        ax.set_ylabel("curvature", color="C1", rotation=270)

        plt.savefig(output[0], bbox_inches="tight", verbose=False)

        """Slope
        """

        d0 = model.basis.order[0]
        with torch.no_grad():

            fpred = model.surface(Zpred)
            lower, upper = fpred.confidence_region()

            lmu, lvar = grad.gradient(model.surface, Zpred, z0, p=0)
            lmu, lvar = lmu[:, d0, 0], lvar[:, d0, d0]
            nrm = torch.distributions.Normal(lmu, lvar.sqrt())

            lmu = lmu.numpy()
            lvar = lvar.numpy()

        fig, ax = plt.subplots(dpi=200, figsize=(3, 2))
        plt.plot(zrng, fpred.mean[:,0], label="$f(z)$")
        plt.fill_between(zrng, lower[:,0], upper[:,0], alpha=0.6)
        plt.xlabel("$z_1$")

        plt.setp([ax.get_yticklabels()], color="C0")
        ax.tick_params(axis="y", color="C0")
        for pos in [
            "left",
        ]:
            plt.setp(ax.spines[pos], color="C0", linewidth=1.0)
        ax.set_ylabel("$f(\mathbf{z}$)", color="C0")

        ax = plt.twinx()
        plt.plot(zrng, lmu, c="C2", label="Laplacian")
        plt.fill_between(zrng, lmu - np.sqrt(lvar)*2, lmu + np.sqrt(lvar)*2, alpha=0.6, color="C2")
        plt.axhline(0, c="C2", ls="--")

        plt.setp([ax.get_yticklabels()], color="C2")
        ax.tick_params(axis="y", color="C2")
        for pos in [
            "right",
        ]:
            plt.setp(ax.spines[pos], color="C2", linewidth=1.0)
        ax.set_ylabel("slope", color="C2", rotation=270)

        plt.savefig(output[1], bbox_inches="tight", verbose=False)
