import pickle

import pandas as pd
import numpy as np
import dms_variants.binarymap
import dms_variants.globalepistasis

rule ge_cv:
    input:
        "data/processed/{ds}.csv"
    output:
        "experiments/{ds}-{phenotype}/globalep/cv{cv}/model.pkl",
        "experiments/{ds}-{phenotype}/globalep/cv{cv}/pred-val.csv"
    run:
        def cget(pth, default=None):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # phenotype = cget("phenotypes")[wildcards.phenotype].get("col", "phenotype")
        phenotype = cget(f"phenotypes/{wildcards.phenotype}", default=["phenotype"])[0]
        # noise = cget("phenotypes")[wildcards.phenotype].get("noise", None)
        noise = cget(f"errors/{wildcards.phenotype}", default=None)
        if isinstance(noise, list):
            noise = noise[0]

        df = pd.read_csv(input[0])
        df = df.assign(
            aa_substitutions=(
                df[cget("substitutions", "substitutions")]
                .replace(np.nan, "")
                .str
                .replace(":", " ")
            ),
            func_score=df[phenotype],
        )

        # hack for avGFP to remove leading "S"
        if wildcards.ds == "gfp":
            df.aa_substitutions = df.aa_substitutions.str.split(" ").apply(
                    lambda x: " ".join([xx[1:] for xx in x])
                )

        if noise is not None:
            df = df.assign(
                func_score_var=df[noise],
            )

        alphabet = cget("alphabet", [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
                "*",
            ]
        )

        bmap = dms_variants.binarymap.BinaryMap(
            df[df.cv != int(wildcards.cv)],
            alphabet=alphabet,
            func_score_var_col="func_score_var" if noise is not None else None,
        )

        ifit_df = dms_variants.globalepistasis.fit_models(bmap, "Gaussian")
        ge = ifit_df.loc[0].model

        val = df[df.cv == int(wildcards.cv)].copy()

        # fix for missing subs
        val["aa_substitutions"] = val["aa_substitutions"].apply(
            lambda x: " ".join([xx for xx in x.split(" ") if xx in ge.binarymap.all_subs])
        )

        pred = ge.add_phenotypes_to_df(val, unknown_as_nan=False)
        pred.dropna(subset=["observed_phenotype"], inplace=True)
        pred.to_csv(
            output[1],
            index=False,
        )

        pickle.dump(ge, open(output[0], "wb"))
