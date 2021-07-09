
rule dimensions:
    input:
        expand(
            "experiments/{ds}/lantern/full/model.pt",
            ds=["gfp-brightness", "laci-joint", "covid-joint"],
        )
    output:
        "figures/dimensions.png"
    run:
        mn = 1e-4
        mx = 1

        def invgammalogpdf(x, alpha, beta):
            return alpha * beta.log() - torch.lgamma(alpha) + (-alpha-1)*x.log() - beta/x

        plt.figure(figsize=(2.5, 3), dpi=300)
        for pth in input:

            ds, phenotype = pth.split("/")[1].split("-")
            lab = config[ds]["label"]

            def dsget(pth, default):
                """Get the configuration for the specific dataset"""
                return get(config, f"{ds}/{pth}", default=default)

            df, ds, model = util.load_run(ds, phenotype, "lantern", "full", dsget("K", 8))
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
