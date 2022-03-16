from torch import nn
from torch.nn import functional as F
import attr


@attr.s(eq=False)
class Linear(nn.Module):

    K: int = attr.ib()
    D: int = attr.ib()

    def __attrs_post_init__(self):
        super(Linear, self).__init__()
        self.layer = nn.Linear(self.K, self.D)

    def _initialize(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self.layer(x)

    def loss(self, x, y, noise=None):

        yhat = self(x)
        loss = F.mse_loss(yhat, y.float(), reduction="none")

        if noise is not None:
            nz = torch.count_nonzero(noise, dim=0)

            # check that noise is not all zero
            for i, nnz in enumerate(nz):
                if nnz > 0:
                    loss[:, i] = loss[:, i] / noise[:, i]


        return loss.mean()

    def _optimizers(self):
        return []

    def optimizationParams(self):
        return [{"params": self.parameters()}]

    def _diagnostic_plots(self, *args, **kwargs):
        return []

    def _additional_logs(self, *args, **kwargs):
        return {}

    def embed(self, x):
        raise ValueError("no embedding learned!")


rule lin_cv:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}/model.pt"
    group: "linear"
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

        # setup model
        model = Linear(ds.p, ds.D)

        # cuda
        if torch.cuda.is_available():
            model = model.cuda()
            ds.to("cuda")

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(train, batch_size=dsget("linear/batch-size", default=128))
        vloader = DataLoader(validation, batch_size=dsget("linear/batch-size", default=128))

        lr = dsget("linear/lr", default=0.001)
        optimizer = Adam(model.parameters(), lr=lr)

        mlflow.set_experiment("linear cross-validation")

        with mlflow.start_run() as run:
            mlflow.log_param("dataset", wildcards.ds)
            mlflow.log_param("model", "linear")
            mlflow.log_param("lr", lr)
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(dsget("linear/epochs", default=50)),)
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


rule lin_prediction:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}/pred-val.csv"
    group: "linear"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        import pickle

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        CUDA = dsget("linear/prediction/cuda", default=False)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # setup model
        model = Linear(ds.p, ds.D)
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
                    size=dsget("linear/prediction/size", default=1024),
                    pbar=dsget("linear/prediction/pbar", default=True)
                )
            )

rule lin_cv_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}-n{n}/model.pt",
    group: "linear"
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

        # setup model
        model = Linear(ds.p, ds.D)

        # cuda
        if torch.cuda.is_available():
            model = model.cuda()
            ds.to("cuda")

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
        tloader = DataLoader(train, batch_size=dsget("linear/batch-size", default=128))
        vloader = DataLoader(validation, batch_size=dsget("linear/batch-size", default=128))

        lr = dsget("linear/lr", default=0.001)
        optimizer = Adam(model.parameters(), lr=lr)

        mlflow.set_experiment(f"linear cross-validation n={wildcards.n}")

        with mlflow.start_run() as run:
            mlflow.log_param("dataset", wildcards.ds)
            mlflow.log_param("model", "linear")
            mlflow.log_param("lr", lr)
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(dsget("linear/epochs", default=50)),)
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

rule lin_prediction_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}-n{n}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/linear/cv{cv,\d+}-n{n}/pred-val.csv"
    group: "linear"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        import pickle

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        CUDA = dsget("linear/prediction/cuda", default=False)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # setup model
        model = Linear(ds.p, ds.D)
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
                    size=dsget("linear/prediction/size", default=1024),
                    pbar=dsget("linear/prediction/pbar", default=True)
                )
            )
