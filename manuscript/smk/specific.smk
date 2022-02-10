
rule specific_epistasis_predictions:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl"
    output:
        "experiments/{ds}-{phenotype}/specific/cv{cv}/model.pkl",
        "experiments/{ds}-{phenotype}/specific/cv{cv}/pred-val.csv"
    group:
        "train"
    resources:
        mem_mb="32000M",
        mem = "32000M",
        time = "4:00:00",
        partition = "batch"
    run:
        import pickle
        from itertools import combinations

        from scipy import sparse
        from sklearn.linear_model import LassoCV
        
        import src.specific

        def cget(pth, default=None):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        dat = ds[:len(ds)]
        X, y = dat[:2]
        Xtrain = X[df.cv != int(wildcards.cv), :]
        ytrain = y[df.cv != int(wildcards.cv), :]

        Xtest = X[df.cv == int(wildcards.cv), :]
        ytest = y[df.cv == int(wildcards.cv), :]

        Xtrain_s, interactions = src.specific.sparse_design(Xtrain)
        Xtest_s, interactions = src.specific.sparse_design(
            Xtest, interactions=interactions
        )

        # normalize b/c no intercept
        mu = ytrain.mean()
        sigma = ytrain.std()
        ytrain = (ytrain - mu) / sigma
        ytest = (ytest - mu) / sigma

        # find best alpha, don't fit intercept b/c it's slow
        # reg = LassoCV(
        #     cv=5, random_state=0, selection="random", fit_intercept=False, n_alphas=15
        # )

        # only penalize interactions
        weight = np.ones(Xtrain_s.shape[1])
        weight[:Xtrain.shape[1]] = 0

        # GroupLasso
        from group_lasso import GroupLasso
        groups = np.ones(Xtrain_s.shape[1])
        groups[:Xtrain.shape[1]] = -1
        reg = GroupLasso(groups=groups, l1_reg=0.0, group_reg=0.05, scale_reg="none")

        from glmnet import ElasticNet
        reg = ElasticNet(n_splits=10, standardize=False)
        groups[:Xtrain.shape[1]] = 0

        # also noise
        if len(dat) > 2:
            n = dat[2]
            ntrain = n[df.cv != int(wildcards.cv), :]
            ntest = n[df.cv == int(wildcards.cv), :]
            # reg = reg.fit(Xtrain_s, ytrain[:, 0].numpy(), sample_weight=1 / ntrain)
            reg = reg.fit(Xtrain_s, ytrain[:, 0].numpy(), relative_penalties=groups, sample_weight=1 / ntrain)
            # beta = src.specific.partial_lasso(Xtrain_s, ytrain[:, 0].numpy(), weight, 0.1, sample_weight=1/ntrain)
        else:
            # reg = reg.fit(Xtrain_s, ytrain[:, 0].numpy())
            reg = reg.fit(Xtrain_s, ytrain[:, 0].numpy(), relative_penalties=groups)
            # beta = src.specific.partial_lasso(Xtrain_s, ytrain[:, 0].numpy(), weight, 0.1)

        pickle.dump(reg, open(output[0], "wb"))

        # make prediction
        yhat = reg.predict(Xtest_s)
        # yhat = np.dot(Xtest_s, beta)

        df = pd.DataFrame(dict(y0=ytest[:, 0].numpy(), yhat0=yhat)).assign(
            cv=wildcards.cv
        )
        if len(dat) > 2:
            df = df.assign(noise0=ntest)
        df.to_csv(output[1])
