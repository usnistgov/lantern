import pickle

import pandas as pd
import numpy as np
import dms_variants.binarymap
import dms_variants.globalepistasis

# data dependencies for all training
subworkflow data:
    snakefile:
        "data.smk"

rule cv:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"]))
    output:
        expand("experiments/{ds}-{phenotype}/globalep/cv{cv}/model.pkl", ds=config["name"], allow_missing=True),
        expand("experiments/{ds}-{phenotype}/globalep/cv{cv}/pred-val.csv", ds=config["name"], allow_missing=True),
    run:

        phenotype = config.get("phenotypes")[wildcards.phenotype].get("col", "phenotype")
        noise = config.get("phenotypes")[wildcards.phenotype].get("noise", None)

        df = pd.read_csv(input[0])
        df = df.assign(
            aa_substitutions=(
                df[config.get("substitutions", "substitutions")]
                .replace(np.nan, "")
                .str
                .replace(":", " ")
            ),
            func_score=df[phenotype],
        )

        # hack for avGFP to remove leading "S"
        if config.get("name") == "gfp":
            df.aa_substitutions = df.aa_substitutions.str.split(" ").apply(
                    lambda x: " ".join([xx[1:] for xx in x])
                )

        if noise is not None:
            df = df.assign(
                func_score_var=df[noise],
            )

        alphabet = config.get("alphabet", [
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
