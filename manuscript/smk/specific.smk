
rule specific_epistasis_predictions:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl"
    output:
        "experiments/{ds}-{phenotype}/specific/cv{cv}/model.pkl",
        "experiments/{ds}-{phenotype}/specific/cv{cv}/pred-val.csv"
    resources:
        mem = "16000M",
        time = "4:00:00"
    run:
        import pickle
        from itertools import combinations

        from scipy import sparse
        from sklearn.linear_model import LassoCV
        
        def cget(pth, default=None):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))
        X, y = ds[:len(ds)][:2]

        Xtrain = X[df.cv != int(wildcards.cv), :]
        ytrain = y[df.cv != int(wildcards.cv), :]

        Xtest = X[df.cv == int(wildcards.cv), :]
        ytest = y[df.cv == int(wildcards.cv), :]

        # build sparse design matrix
        # first, get individual effects
        I, J = np.where(Xtrain.numpy())

        # need to add more
        II = I.tolist()
        JJ = J.tolist()

        # now add second and third order interactions
        interactions = {}
        for n in np.unique(I):
            muts = J[I == n]

            for combo in combinations(muts, 2):
                if combo not in interactions:
                    interactions[combo] = len(interactions)
                II.append(n)
                JJ.append(interactions[combo] + X.shape[1])

        print(len(interactions))

        for n in np.unique(I):
            muts = J[I == n]

            for combo in combinations(muts, 3):
                if combo not in interactions:
                    interactions[combo] = len(interactions)
                II.append(n)
                JJ.append(interactions[combo] + X.shape[1])

        print(len(interactions))

        I = np.array(II)
        J = np.array(JJ)
        D = np.ones(I.shape)

        Xtrain_s = sparse.coo_matrix(
            (D, (I, J)), shape=[Xtrain.shape[0], len(interactions) + X.shape[1]]
        ).tocsc()

        # find best alpha
        reg = LassoCV(
            cv=5, random_state=0, selection="random", fit_intercept=False, n_alphas=30
        ).fit(Xtrain_s, ytrain[:, 0].numpy())
        pickle.dump(reg, open(output[0], "wb"))

        # make predictions
        I, J = np.where(Xtest.numpy())

        # need to add more
        II = I.tolist()
        JJ = J.tolist()

        # now add second and third order interactions
        for n in np.unique(I):
            muts = J[I == n]

            for combo in combinations(muts, 2):
                # can't add this one now
                if combo not in interactions:
                    continue
                II.append(n)
                JJ.append(interactions[combo] + X.shape[1])

        print(len(interactions))

        for n in np.unique(I):
            muts = J[I == n]

            for combo in combinations(muts, 3):
                # can't add this one now
                if combo not in interactions:
                    continue
                II.append(n)
                JJ.append(interactions[combo] + X.shape[1])

        print(len(interactions))

        I = np.array(II)
        J = np.array(JJ)
        D = np.ones(I.shape)

        Xtest_s = sparse.coo_matrix(
            (D, (I, J)), shape=[Xtest.shape[0], len(interactions) + X.shape[1]]
        ).tocsc()
        yhat = reg.predict(Xtest_s)

        pd.DataFrame(dict(y0=ytest[:, 0].numpy(), yhat0=yhat)).assign(
            cv=wildcards.cv
        ).to_csv(output[1])
