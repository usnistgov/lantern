from functools import partial
import pickle
import os
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Subset
from dpath.util import get
from tqdm import tqdm
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import DataLoader

from lantern.dataset import Dataset
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype

from src.predict import predictions

def train_lantern(
    input, output, wildcards, tloader, vloader, optimizer, loss, model, lr, epochs
):
    """General purpose lantern training optimization loop
    """

    try:
        with mlflow.start_run() as run:
            mlflow.log_param("dataset", wildcards.ds)
            mlflow.log_param("model", "lantern")
            mlflow.log_param("lr", lr)
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(epochs),)
            record = defaultdict(list)
            for e in pbar:

                # logging of loss values
                tloss = 0
                vloss = 0

                # store total loss terms
                terms = defaultdict(list)

                # go through minibatches
                for btch in tloader:
                    optimizer.zero_grad()
                    yhat = model(btch[0])
                    lss = loss(yhat, *btch[1:])

                    for k, v in lss.items():
                        terms["{}-train".format(k)].append(v.item())

                    total = sum(lss.values())
                    total.backward()

                    optimizer.step()
                    tloss += total.item()

                # validation minibatches
                for btch in vloader:
                    with torch.no_grad():
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        for k, v in lss.items():
                            terms["{}-validation".format(k)].append(v.item())

                        total = sum(lss.values())
                        vloss += total.item()

                # update record
                record["epoch"].append(e)
                for k, v in terms.items():
                    record[k].append(np.mean(v))

                # update log
                pbar.set_postfix(
                    train=tloss / len(tloader),
                    validation=vloss / len(vloader) if len(vloader) else 0,
                )

                mlflow.log_metric("training-loss", tloss / len(tloader), step=e)

                if len(vloader):
                    mlflow.log_metric("validation-loss", vloss / len(vloader), step=e)

                qalpha = model.basis.qalpha(detach=True)
                for k in range(model.basis.K):
                    mlflow.log_metric(
                        f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e
                    )
                    mlflow.log_metric(
                        f"basis-log-alpha-{k}",
                        model.basis.log_alpha[k].detach().item(),
                        step=e,
                    )
                    mlflow.log_metric(
                        f"basis-log-beta-{k}",
                        model.basis.log_beta[k].detach().item(),
                        step=e,
                    )

            # Save training results
            torch.save(model.state_dict(), output[0])
            torch.save(loss.state_dict(), output[1])

            # also save this specific version by id
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model.pt"))
            mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
            torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
            mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

    except Exception as e:
        base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
        os.makedirs(base, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
        torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
        raise e

    return pd.DataFrame(record)

rule lantern_cv:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/loss.pt"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8), meanEffectsInit=False),
            Phenotype.fromDataset(ds, dsget("K", 8))
        )

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(train, batch_size=dsget("lantern/batch-size", default=8192))
        vloader = DataLoader(validation, batch_size=dsget("lantern/batch-size", default=8192))

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment("lantern cross-validation")

        # Run optimization
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("dataset", wildcards.ds)
                mlflow.log_param("model", "lantern")
                mlflow.log_param("lr", dsget("lantern/lr", default=0.01))
                mlflow.log_param("cv", wildcards.cv)
                mlflow.log_param("batch-size", tloader.batch_size)

                pbar = tqdm(range(dsget("lantern/epochs", default=5000)),)
                best = np.inf
                for e in pbar:

                    # logging of loss values
                    tloss = 0
                    vloss = 0

                    # go through minibatches
                    for btch in tloader:
                        optimizer.zero_grad()
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        total = sum(lss.values())
                        total.backward()

                        optimizer.step()
                        tloss += total.item()

                    # validation minibatches
                    for btch in vloader:
                        with torch.no_grad():
                            yhat = model(btch[0])
                            lss = loss(yhat, *btch[1:])

                            total = sum(lss.values())
                            vloss += total.item()

                    # update log
                    pbar.set_postfix(
                        train=tloss / len(tloader),
                        validation=vloss / len(vloader) if len(vloader) else 0,
                    )

                    mlflow.log_metric("training-loss", tloss / len(tloader), step=e)
                    mlflow.log_metric("validation-loss", vloss / len(vloader), step=e)

                    qalpha = model.basis.qalpha(detach=True)
                    for k in range(model.basis.K):
                        mlflow.log_metric(f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e)

                # Save training results
                torch.save(model.state_dict(), output[0])
                torch.save(loss.state_dict(), output[1])

                # also save this specific version by id
                base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
                os.makedirs(base, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(base, "model.pt"))
                mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
                torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
                mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

        except Exception as e:
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
            torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
            raise e

rule lantern_full:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/full{rerun,-r.*}{kernel,-kern-.*}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/full{rerun,-r.*}{kernel,-kern-.*}/loss.pt"
    resources:
        gres="gpu:1",
        partition="batch",
    group: "train"
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        # Setup training infrastructure
        tloader = DataLoader(ds, batch_size=dsget("lantern/batch-size", default=8192))

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment("lantern full")

        # Run optimization
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("dataset", wildcards.ds)
                mlflow.log_param("model", "lantern")
                mlflow.log_param("lr", dsget("lantern/lr", default=0.01))
                mlflow.log_param("batch-size", tloader.batch_size)

                pbar = tqdm(range(dsget("lantern/epochs", default=5000)),)
                best = np.inf
                for e in pbar:

                    # logging of loss values
                    tloss = 0

                    # go through minibatches
                    for btch in tloader:
                        optimizer.zero_grad()
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        total = sum(lss.values())
                        total.backward()

                        optimizer.step()
                        tloss += total.item()

                    # update log
                    pbar.set_postfix(train=tloss / len(tloader),)

                    mlflow.log_metric("training-loss", tloss / len(tloader), step=e)

                    qalpha = model.basis.qalpha(detach=True)
                    for k in range(model.basis.K):
                        mlflow.log_metric(
                            f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e
                        )

                # Save training results
                torch.save(model.state_dict(), output[0])
                torch.save(loss.state_dict(), output[1])

                # also save this specific version by id
                base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
                os.makedirs(base, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(base, "model.pt"))
                mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
                torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
                mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

        except Exception as e:
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
            torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
            raise e

rule lantern_prediction:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )

rule lantern_cv_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/loss.pt"
    group: "train"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8), meanEffectsInit=False),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        # Setup training infrastructure
        train = Subset(
            ds,
            np.random.choice(
                np.where(df.cv != float(wildcards.cv))[0],
                int(wildcards.n),
                replace=False,
            ),
        )
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(
            train, batch_size=dsget("lantern/batch-size", default=8192)
        )
        vloader = DataLoader(
            validation, batch_size=dsget("lantern/batch-size", default=8192)
        )

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment(f"lantern cross-validation n={wildcards.n}")

        lr = dsget("lantern/lr", default=0.01)
        epochs = dsget("lantern/epochs", default=5000)

        train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs,
        )

rule lantern_prediction_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )

rule lantern_cv_k:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/model.pt",
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/loss.pt"
    resources:
        gres="gpu:1",
        partition="singlegpu",
        time = "24:00:00",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, int(wildcards.k), meanEffectsInit=False),
            Phenotype.fromDataset(ds, int(wildcards.k))
        )

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(
            train, batch_size=dsget("lantern/batch-size", default=8192)
        )
        vloader = DataLoader(
            validation, batch_size=dsget("lantern/batch-size", default=8192)
        )

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment(f"lantern cross-validation K={wildcards.k}")

        lr = dsget("lantern/lr", default=0.01)
        epochs = dsget("lantern/epochs", default=5000)

        train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs,
        )


rule lantern_prediction_cv_k:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="singlegpu",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, int(wildcards.k), meanEffectsInit=False),
            Phenotype.fromDataset(ds, int(wildcards.k))
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )

rule lantern_affine:
    input:
        csv="data/processed/{ds}.csv",
        ds="data/processed/{ds}-{phenotype}.pkl",
        model="experiments/{ds}-{phenotype}/lantern/full/model.pt",
        # loss="experiments/{ds}-{phenotype}/lantern/full/loss.pt",
    output:
        model="experiments/{ds}-{phenotype}/lantern/affine/{label}/model.pt",
        loss="experiments/{ds}-{phenotype}/lantern/affine/{label}/loss.pt",
        record="experiments/{ds}-{phenotype}/lantern/affine/{label}/history.csv",
    resources:
        gres="gpu:1",
        partition="singlegpu",
        time = "24:00:00",
    run:
        from src import affine
        
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8), meanEffectsInit=False),
            Phenotype.fromDataset(ds, dsget("K", 8))
        )

        # Setup training infrastructure
        tloader = DataLoader(ds, batch_size=dsget("lantern/batch-size", default=8192))

        # empty validation, just to match interface
        validation = Subset(ds, np.where(df.cv == -100)[0])
        vloader = DataLoader(
            validation, batch_size=dsget("lantern/batch-size", default=8192)
        )

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)

        # pre-load
        model.load_state_dict(
            torch.load(input.model, "cpu")
        )
        # loss.load_state_dict(
        #     torch.load(input.loss, "cpu")
        # )

        # apply affine tranform
        tcfgs = dsget(f"lantern/affine/{wildcards.label}/transformations", [])
        transforms = []
        i, j = model.basis.order[:2] # always on top two dimensions
        for tc in tcfgs:
            if tc["transform"] == "rotation":
                transforms.append(
                    affine.Rotation(model.basis.K, i, j, torch.tensor(tc["theta"]))
                )
            elif tc["transform"] == "scale":
                transforms.append(
                    affine.Scale(model.basis.K, i, j, torch.tensor(tc["si"]), torch.tensor(tc["sj"]))
                )
            elif tc["transform"] == "shear":
                transforms.append(
                    affine.Shear(model.basis.K, i, j, torch.tensor(tc["si"]), torch.tensor(tc["sj"]))
                )
            else:
                raise ValueError("Unknown transform {}".format(tc["transform"]))
        affine.transform(model, *transforms)

        ##############################################################################
        # plot resulting surface
        model.eval()

        targets = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}").keys()
        for targ in targets:

            alpha = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/alpha",
                default=0.01,
            )
            raw = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/raw",
                default=None,
            )
            log = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/log",
                default=False,
            )
            p = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/p",
                default=0,
            )
            image = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/image",
                default=False,
            )
            scatter = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/scatter",
                default=True,
            )
            mask = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/mask",
                default=False,
            )
            cbar_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/cbar_kwargs",
                default={},
            )
            fig_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/fig_kwargs",
                default=dict(dpi=300, figsize=(4, 3)),
            )
            cbar_title = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/cbar_title",
                default=None,
            )
            plot_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/plot_kwargs",
                default={},
            )

            z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
                model,
                ds,
                mu=df[raw].mean() if raw is not None else 0,
                std=df[raw].std() if raw is not None else 1,
                log=log,
                p=p,
                alpha=alpha,
            )

            fig, norm, cmap, vrange = util.plotLandscape(
                z,
                fmu,
                fvar,
                Z1,
                Z2,
                log=log,
                image=image,
                mask=mask,
                cbar_kwargs=cbar_kwargs,
                fig_kwargs=fig_kwargs,
                **plot_kwargs
            )

            if scatter:

                plt.scatter(
                    z[:, 0],
                    z[:, 1],
                    c=y,
                    alpha=0.4,
                    rasterized=True,
                    vmin=vrange[0],
                    vmax=vrange[1],
                    norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
                    s=0.3,
                )

                # reset limits
                plt.xlim(
                    np.quantile(z[:, 0], alpha / 2),
                    np.quantile(z[:, 0], 1 - alpha / 2),
                )
                plt.ylim(
                    np.quantile(z[:, 1], alpha / 2),
                    np.quantile(z[:, 1], 1 - alpha / 2),
                )

            if cbar_title is not None:
                fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

            os.makedirs(f"figures/{wildcards.ds}-{wildcards.phenotype}/affine/{wildcards.label}/", exist_ok=True)
            plt.savefig(
                f"figures/{wildcards.ds}-{wildcards.phenotype}/affine/{wildcards.label}/surface-{targ}-init.png",
                bbox_inches="tight",
            )

        model.train()
        # end surface plot
        ##############################################################################

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        mlflow.set_experiment(f"lantern affine")

        ##############################################################################
        # pretrain liklihood

        loss = model.surface.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        if dsget("cuda", True):
            loss = loss.cuda()

        prms = list(loss.mll.likelihood.parameters())
        if ds.errors is not None:
            prms.append(loss.raw_sigma_hoc)

        optimizer = Adam(prms, lr=dsget("lantern/lr", default=0.01))

        lr = dsget("lantern/lr", default=0.01)

        wildcards.cv = -1
        record = train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs=500,
        )

        ploss = loss
        
        ##############################################################################

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)
        if dsget("cuda", True):
            loss = loss.cuda()

        loss.losses[-1].load_state_dict(ploss.state_dict())

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        lr = dsget("lantern/lr", default=0.01)
        epochs = dsget("lantern/epochs", default=5000)

        wildcards.cv = -1
        record = train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs,
        )

        record.to_csv(output.record)

        ##############################################################################
        # plot resulting surface
        if dsget("cuda", True):
            model = model.cpu()
            loss = loss.cpu()
            ds.to("cpu")

        model.eval()

        targets = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}").keys()
        for targ in targets:

            alpha = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/alpha",
                default=0.01,
            )
            raw = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/raw",
                default=None,
            )
            log = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/log",
                default=False,
            )
            p = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/p",
                default=0,
            )
            image = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/image",
                default=False,
            )
            scatter = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/scatter",
                default=True,
            )
            mask = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/mask",
                default=False,
            )
            cbar_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/cbar_kwargs",
                default={},
            )
            fig_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/fig_kwargs",
                default=dict(dpi=300, figsize=(4, 3)),
            )
            cbar_title = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/cbar_title",
                default=None,
            )
            plot_kwargs = get(
                config,
                f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{targ}/plot_kwargs",
                default={},
            )

            z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
                model,
                ds,
                mu=df[raw].mean() if raw is not None else 0,
                std=df[raw].std() if raw is not None else 1,
                log=log,
                p=p,
                alpha=alpha,
            )

            fig, norm, cmap, vrange = util.plotLandscape(
                z,
                fmu,
                fvar,
                Z1,
                Z2,
                log=log,
                image=image,
                mask=mask,
                cbar_kwargs=cbar_kwargs,
                fig_kwargs=fig_kwargs,
                **plot_kwargs
            )

            if scatter:

                plt.scatter(
                    z[:, 0],
                    z[:, 1],
                    c=y,
                    alpha=0.4,
                    rasterized=True,
                    vmin=vrange[0],
                    vmax=vrange[1],
                    norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
                    s=0.3,
                )

                # reset limits
                plt.xlim(
                    np.quantile(z[:, 0], alpha / 2),
                    np.quantile(z[:, 0], 1 - alpha / 2),
                )
                plt.ylim(
                    np.quantile(z[:, 1], alpha / 2),
                    np.quantile(z[:, 1], 1 - alpha / 2),
                )

            if cbar_title is not None:
                fig.axes[-1].set_title(cbar_title, y=1.04, loc="left", ha="left")

            os.makedirs(f"figures/{wildcards.ds}-{wildcards.phenotype}/affine/{wildcards.label}/", exist_ok=True)
            plt.savefig(
                f"figures/{wildcards.ds}-{wildcards.phenotype}/affine/{wildcards.label}/surface-{targ}-final.png",
                bbox_inches="tight",
            )

        model.train()
        # end surface plot
        ##############################################################################
