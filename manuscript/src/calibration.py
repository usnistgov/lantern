import torch
from scipy.stats import norm
import pandas as pd
import numpy as np


def noises(loss, parameters):
    noises = []
    with torch.no_grad():
        for pt in parameters:
            loss.load_state_dict(torch.load(pt, "cpu"))
            likelihood = loss.losses[1].mll.likelihood

            if hasattr(likelihood, "task_noises"):
                noises.append((likelihood.noise + likelihood.task_noises).numpy())
            else:
                noises.append(likelihood.noise.numpy())

    return noises


def prediction_cdf(D, predictions, noises):

    scores = None
    for pth, nz in zip(predictions, noises):
        tmp = pd.read_csv(pth)

        ydist = norm(
            tmp[[f"y{d}" for d in range(D)]].values
            - tmp[[f"yhat{d}" for d in range(D)]].values,
            (tmp[[f"yhat_std{d}" for d in range(D)]] ** 2 + nz[None, :]) ** 0.5,
        )

        tmp[[f"y{d}-cdf0" for d in range(D)]] = ydist.cdf(0.0)

        if scores is None:
            scores = tmp
        else:
            scores = pd.concat((scores, tmp))

    return scores


def predictive_distribution(D, predictions, noise, centered=False):

    tmp = pd.read_csv(predictions)

    if centered:
        mu = (
            tmp[[f"y{d}" for d in range(D)]].values
            - tmp[[f"yhat{d}" for d in range(D)]].values
        )
    else:
        mu = tmp[[f"yhat{d}" for d in range(D)]].values

    ydist = norm(
        mu, (tmp[[f"yhat_std{d}" for d in range(D)]] ** 2 + noise[None, :]) ** 0.5,
    )

    return ydist


def balanced_sample(y, N=100, bins=100):

    h, bins = np.histogram(y, bins)

    # probability of choosing a bin
    weight = np.nan_to_num(1 / np.copy(h), posinf=0)
    weight = weight / weight.sum()

    # bin assignment
    ybin = np.digitize(y, bins, right=True)

    # update to skip out of bounds index
    ybin = np.where(ybin < weight.shape[0], ybin, weight.shape[0] - 1)

    # re-normalize
    yweight = weight[ybin]
    yweight = yweight / yweight.sum()

    # counts
    ycount = h[ybin]

    return (
        np.random.choice(np.arange(y.shape[0]), N, replace=False, p=yweight),
        yweight,
        ycount,
    )
