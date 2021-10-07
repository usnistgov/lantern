import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker

from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype


def load_run(dataset, phenotype, model, run, K=8):
    df = pd.read_csv(f"data/processed/{dataset}.csv")
    ds = pickle.load(open(f"data/processed/{dataset}-{phenotype}.pkl", "rb"))

    # Build model and loss
    _model = Model(VariationalBasis.fromDataset(ds, K), Phenotype.fromDataset(ds, K),)

    _model.load_state_dict(
        torch.load(f"experiments/{dataset}-{phenotype}/{model}/{run}/model.pt", "cpu")
    )

    return df, ds, _model


def buildLandscape(
    model,
    ds,
    d0=0,
    d1=1,
    mu=0,
    std=1,
    offset=0,
    log=False,
    p=0,
    fixed=None,
    alpha=0,
    alpha1=0,
    alpha2=0,
    N=100,
    lim=None,
):

    i1, i2 = model.basis.order[[d0, d1]]

    X, y = ds[: len(ds)][:2]
    y = y[:, p].numpy()
    with torch.no_grad():
        z = model.basis(X)
        z = z[:, model.basis.order]

    # build from dataset
    if lim is None:
        Z1 = np.linspace(
            np.quantile(z[:, d0], alpha / 2), np.quantile(z[:, d0], 1 - alpha / 2), N,
        )
        Z2 = np.linspace(
            # np.quantile(z[:, d1], alpha / 2), np.quantile(z[:, d1], 1 - alpha / 2), N,
            np.quantile(z[:, d1], 1 - alpha / 2),
            np.quantile(z[:, d1], alpha / 2),
            N,
        )
    else:
        Z1 = np.linspace(*lim[:2], N)
        Z2 = np.linspace(*lim[2:], N)

    Z1, Z2 = np.meshgrid(Z1, Z2)

    Z = torch.zeros(N ** 2, model.basis.K)

    Z[:, i1] = torch.from_numpy(Z1.ravel())
    Z[:, i2] = torch.from_numpy(Z2.ravel())

    # add fixed values
    if fixed is not None:
        for i, fx in zip(model.basis.order, fixed):
            if fx is None:
                continue

            Z[:, i] = fx

    with torch.no_grad():
        f = model.surface(Z)
        fmu = f.mean.numpy()
        fvar = f.variance.numpy()

    mean = model.surface.mean.constant
    ops = model.surface.kernel.outputscale
    if fmu.ndim > 1:
        fmu = fmu[:, p]
        fvar = fvar[:, p]
        ops = ops[p]
        mean = mean[p]
    ops = ops.item()

    fmu = (fmu + offset) * std + mu
    y = (y + offset) * std + mu

    # scale to base variance
    fvar = fvar / ops

    if log:
        fmu = np.power(10, fmu)
        y = np.power(10, y)

    return z, fmu, fvar, Z1, Z2, y, Z


def plotLandscape(
    z,
    fmu,
    fvar,
    Z1,
    Z2,
    log=False,
    maxshade=0.6,
    d0=0,
    d1=1,
    C0="k",
    C1="k",
    contour_label="",
    inducing=None,
    varlog=False,
    plotOrigin=True,
    fig=None,
    ax=None,
    image=True,
    mask=True,
    vrange=None,
    colorbar=True,
    log_tick_subs=[0.5, 1.0],
    levels=8,
    contour_kwargs={},
    varColor=False,
    cbar_kwargs={},
    fig_kwargs=dict(dpi=200, figsize=(6, 4)),
):

    # store for boolean check later
    _mask = mask

    weights = fmu.reshape(Z1.shape)

    if vrange is None:
        vrange = (weights.min(), weights.max())
    else:
        vrange = (min(vrange[0], weights.min()), max(vrange[1], weights.max()))

    mask = np.zeros((*weights.shape, 4))
    mask[:, :, :3] = np.full((*weights.shape, 3), 0, dtype=np.uint8)

    relvar = fvar.reshape(Z1.shape)
    if varlog:
        relvar = np.log(relvar) + 1

    # mask[:, :, 3] = np.minimum(maxshade, relvar - relvar.min())
    mask[:, :, 3] = (relvar - relvar.min()) / (1 - relvar.min()) * maxshade

    if fig is None or ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    if image:
        ax.imshow(
            weights,
            alpha=0.9,
            # norm=mpl.colors.LogNorm(vmin=weights.min(), vmax=weights.max(),)
            norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
            interpolation="lanczos",
            # extent=(z[:, d0].min(), z[:, d0].max(), z[:, d1].min(), z[:, d1].max()),
            extent=(Z1.min(), Z1.max(), Z2.min(), Z2.max()),
            origin="upper",
            aspect="auto",
        )

    if varColor:
        ax.imshow(
            (relvar - relvar.min()) / (1 - relvar.min()),
            extent=(Z1.min(), Z1.max(), Z2.min(), Z2.max()),
            origin="upper",
            aspect="auto",
            cmap="Greys",
            interpolation="lanczos",
        )

    im = ax.contour(
        Z1,
        Z2,
        weights,
        levels=levels,
        vmin=vrange[0],
        vmax=vrange[1],
        # vmin=weights.min(),
        # vmax=weights.max(),
        locator=ticker.LogLocator(subs=log_tick_subs) if log else None,
        **contour_kwargs,
    )

    if _mask:
        ax.imshow(
            mask,
            zorder=100,
            interpolation="lanczos",
            # extent=(z[:, d0].min(), z[:, d0].max(), z[:, d1].min(), z[:, d1].max()),
            extent=(Z1.min(), Z1.max(), Z2.min(), Z2.max()),
            origin="upper",
            aspect="auto",
        )

    if plotOrigin:
        ax.scatter(0, 0, c="r", marker="x", zorder=10000)
    if inducing is not None:
        ax.scatter(inducing[:, 0], inducing[:, 1], zorder=10000, alpha=0.3, c="orange")

    # color z1
    plt.setp([ax.get_xticklabels()], color=C0)
    ax.tick_params(axis="x", color=C0)
    for pos in [
        "bottom",
    ]:
        plt.setp(ax.spines[pos], color=C0, linewidth=1.5)
    ax.set_xlabel("$z_{}$".format(d0 + 1), color=C0)

    # color z2
    plt.setp([ax.get_yticklabels()], color=C1)
    ax.tick_params(axis="y", color=C1)
    for pos in [
        "left",
    ]:
        plt.setp(ax.spines[pos], color=C1, linewidth=1.5)
    ax.set_ylabel("$z_{}$".format(d1 + 1), color=C1)

    # Set background color
    # ax.set_facecolor(bkgcolor)

    # if hist:
    #     cbar = fig.colorbar(hist[3], ax=fig.axes, anchor=(0.325, 0))
    #     cbar.ax.set_ylabel("Variant count")
    #     ax = fig.axes[:-1]

    cs = im

    if log:
        # norm = mpl.colors.LogNorm(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        norm = mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1])
    else:
        # norm = mpl.colors.Normalize(vmin=fmu.min(), vmax=fmu.max())
        norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array(np.array([]))

    if colorbar:
        cbar = fig.colorbar(sm, ax=ax, **cbar_kwargs)
        cbar.ax.set_ylabel(contour_label)

    return fig, norm, cs.cmap, vrange
