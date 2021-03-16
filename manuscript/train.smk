import os

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
        import torch
        from torch.optim import Adam
        from torch.utils.data import Subset
        # from torch.utils.tensorboard import SummaryWriter
        from dpath.util import get
        from tqdm import tqdm
        import numpy as np
        import pandas as pd
        
        from lantern.dataset import Dataset
        from lantern.model import Model
        from lantern.model.basis import VariationalBasis
        from lantern.model.surface import Phenotype
        from torch.utils.data import DataLoader

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

        # Run optimization
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

            # store loss values
            # writer.add_scalar("Loss/train", tloss / len(tloader), e)
            # writer.add_scalar(
            #     "Loss/validation", vloss / len(vloader) if len(vloader) else 0, e
            # )

            # if self.diagnosticRate > 0 and e % self.diagnosticRate == 0:
            #     torch.save(
            #         self.model.state_dict(),
            #         os.path.join(output, "model-epoch{}.pt".format(e)),
            #     )

            # self._loss["train"].append(tloss / len(tloader))
            # self._loss["validation"].append(vloss / len(vloader) if len(vloader) else 0)

            # # store any loss components if available
            # for k, v in dtloss.items():
            #     self._loss["{}-train".format(k)].append(np.mean(v))
            #     writer.add_scalar("{}/train".format(k), np.mean(v), e)
            # for k, v in dvloss.items():
            #     self._loss["{}-validation".format(k)].append(np.mean(v))
            #     writer.add_scalar("{}/validation".format(k), np.mean(v), e)


        # Save training results
        torch.save(model.state_dict(), output[0])
        torch.save(loss.state_dict(), output[1])
