import os
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

# Feedforward neural network model
class Feedforward(nn.Module):

    def __init__(self, p, K, D, depth=1, width=32):
        super(Feedforward, self).__init__()

        # always need at least these many layers
        layers = [
            nn.Linear(p, K),
            nn.ReLU(),
            nn.Linear(K, width),
            nn.ReLU(),
        ]

        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, D))

        self.layers = nn.ModuleList(
            layers
        )

    def forward(self, x):

        for l in self.layers:
            x = l(x)

        return x

    def loss(self, x, y, noise=None):

        yhat = self(x)
        loss = F.mse_loss(yhat, y.float(), reduction="none")

        if noise is not None:
            loss = loss / noise

        return loss.mean()

# data dependencies for all training
subworkflow data:
    snakefile:
        "data.smk"

rule cv_lantern:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"]))
    output:
        expand("experiments/{ds}/lantern/cv{cv}/model.pt", ds=config["name"], allow_missing=True),
        expand("experiments/{ds}/lantern/cv{cv}/loss.pt", ds=config["name"], allow_missing=True)
    run:
        # Load the dataset
        df = pd.read_csv(input[0])
        ds = Dataset(
            df,
            substitutions=config.get("substitutions", "substitutions"),
            phenotypes=config.get("phenotypes", ["phenotype"]),
            errors=config.get("errors", None),
        )

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, config.get("K", 8)),
            Phenotype.fromDataset(ds, config.get("K", 8))
        )

        loss = model.loss(N=len(ds))

        if config.get("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(train, batch_size=get(config, "lantern/batch-size", default=8192))
        vloader = DataLoader(validation, batch_size=get(config, "lantern/batch-size", default=8192))

        optimizer = Adam(loss.parameters(), lr=get(config, "lantern/lr", default=0.01))
        # writer = SummaryWriter(logdir)

        # mlflow.set_experiment("lantern cross-validation for {label}".format(label=config["label"]))
        mlflow.set_experiment("lantern cross-validation".format(label=config["label"]))

        # Run optimization
        with mlflow.start_run() as run:
            mlflow.log_param("dataset", config["name"])
            mlflow.log_param("model", "lantern")
            mlflow.log_param("lr", get(config, "lantern/lr", default=0.01))
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(get(config, "lantern/epochs", default=5000)),)
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

rule cv_ff:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"]))
    output:
        expand("experiments/{ds}/feedforward-K{K,\d+}/cv{cv}/model.pt", ds=config["name"], allow_missing=True),
    run:
        # Load the dataset
        df = pd.read_csv(input[0])
        ds = Dataset(
            df,
            substitutions=config.get("substitutions", "substitutions"),
            phenotypes=config.get("phenotypes", ["phenotype"]),
            errors=config.get("errors", None),
        )

        # Build model and loss
        DEPTH = get(config, "feedforward/depth", default=1)
        WIDTH = get(config, "feedforward/width", default=32)
        model = Feedforward(
            p=ds.p, K=int(wildcards.K), D=ds.D,
            depth=DEPTH,
            width=WIDTH,
        )

        if config.get("cuda", True):
            model = model.cuda()
            ds.to("cuda")

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(train, batch_size=get(config, "lantern/batch-size", default=128))
        vloader = DataLoader(validation, batch_size=get(config, "lantern/batch-size", default=128))

        lr = get(config, "lantern/lr", default=0.001)
        optimizer = Adam(model.parameters(), lr=lr)

        mlflow.set_experiment("feedforward cross-validation".format(label=config["label"]))

        # Run optimization
        with mlflow.start_run() as run:
            mlflow.log_param("dataset", config["name"])
            mlflow.log_param("model", "lantern")
            mlflow.log_param("lr", lr)
            mlflow.log_param("depth", DEPTH)
            mlflow.log_param("width", WIDTH)
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(get(config, "lantern/epochs", default=100)),)
            best = np.inf
            for e in pbar:

                # logging of loss values
                tloss = 0
                vloss = 0

                # go through minibatches
                for btch in tloader:
                    optimizer.zero_grad()
                    lss = model.loss(*btch)

                    lss.backward()

                    optimizer.step()
                    tloss += lss.item()

                # validation minibatches
                for btch in vloader:
                    with torch.no_grad():
                        lss = model.loss(*btch)

                        vloss += lss.item()

                # update log
                pbar.set_postfix(
                    train=tloss / len(tloader),
                    validation=vloss / len(vloader) if len(vloader) else 0,
                )

                mlflow.log_metric("training-loss", tloss / len(tloader), step=e)
                mlflow.log_metric("validation-loss", vloss / len(vloader), step=e)

            # Save training results
            torch.save(model.state_dict(), output[0])

            # also save this specific version by id
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model.pt"))
            mlflow.log_artifact(os.path.join(base, "model.pt"), "model")

rule pred_lantern:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"])),
        expand("experiments/{ds}/lantern/cv{cv}/model.pt", ds=config["name"], allow_missing=True),
    output:
        expand("experiments/{ds}/lantern/cv{cv}/pred-val.csv", ds=config["name"], allow_missing=True),
    run:
        CUDA = get(config, "lantern/prediction/cuda", default=True)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = Dataset(
            df,
            substitutions=config.get("substitutions", "substitutions"),
            phenotypes=config.get("phenotypes", ["phenotype"]),
            errors=config.get("errors", None),
        )
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, config.get("K", 8)),
            Phenotype.fromDataset(ds, config.get("K", 8))
        )

        model.load_state_dict(torch.load(input[1]))
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
                    size=get(config, "lantern/prediction/size", default=32),
                    pbar=get(config, "lantern/prediction/pbar", default=True)
                )
            )
