
rule gfp_raw:
    output:
        directory("data/raw/gfp"),
        "data/raw/gfp/raw.tsv"
    shell:
        """
        mkdir -p data/raw/gfp
        wget -O data/raw/gfp/raw.zip "https://ndownloader.figshare.com/articles/3102154/versions/1"
        cd data/raw/gfp
        unzip raw.zip
        cp amino_acid_genotypes_to_brightness.tsv raw.tsv
        """

rule gfp:
    input:
        "data/raw/gfp/raw.tsv"
    output:
        "data/processed/gfp.csv"
    run:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import KFold

        # load raw dataset and add needed columns
        raw = pd.read_csv("data/raw/gfp/raw.tsv", sep="\t")
        raw["substitutions"] = raw["aaMutations"]
        raw["phenotype"] = (raw["medianBrightness"] - raw["medianBrightness"].mean()) / raw["medianBrightness"].std()

        # setup CV splits
        cv = KFold(10, shuffle=True, random_state=19033851)
        raw["cv"] = np.nan
        for i, (_, cvts) in enumerate(cv.split(np.arange(len(raw)))):
            raw["cv"].iloc[cvts] = i

        # save
        raw.to_csv("data/processed/gfp.csv")

rule gfp_pkl:
    input:
        "data/processed/gfp.csv"
    output:
        "data/processed/gfp.pkl"
    run:
        import pickle
        import pandas as pd
        from lantern.dataset import Dataset

        df = pd.read_csv(input[0])
        ds = Dataset(
            df,
            substitutions="substitutions",
            phenotypes=["phenotype"],
            errors=None
        )

        pickle.dump(ds, open(output[0], "wb"))


rule ds_pkl:
    "Generate a pickled Dataset object for corresponding dataset and phenotype."

    input:
        "data/processed/{ds}.csv"
    output:
        "data/processed/{ds}-{phenotype}.pkl"
    run:
        import pickle
        import pandas as pd
        from lantern.dataset import Dataset

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        df = pd.read_csv(input[0])
        ds = Dataset(
            df,
            substitutions=dsget("substitutions", default="substitutions"),
            phenotypes=dsget(f"phenotypes/{wildcards.phenotype}", default=["phenotype"]),
            errors=dsget(f"errors/{wildcards.phenotype}", None),
        )

        pickle.dump(ds, open(output[0], "wb"))

rule laci_data:
    input:
        "data/raw/laci.hdf"
    output:
        "data/processed/laci.csv"
    run:
        import pandas as pd
        import numpy as np
        from Bio.Seq import Seq
        from sklearn.model_selection import KFold

        wt = "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQSLLIGVATSSLALHAPSQIVAAIKSRADQLGASVVVSMVERSGVEACKAAVHNLLAQRVSGLIINYPLDDQDAIAVEAACTNVPALFLDVSDQTPINSIIFSHEDGTRLGVEHLVALGHQQIALLAGPLSSVSARLRLAGWHKYLTRNQIQPIAEREGDWSAMSGFQQTMQMLNEGIVPTAMLVANDQMALGAMRAITESGLRVGADISVVGYDDTEDSSCYIPPLTTIKQDFRLLGQTSVDRLLQLSQGQAVKGNQLLPVSLVKRKTTLAPNTQTASPRALADSLMQLARQVSRLESGQ"


        def hamming(s1):
            ret = 0
            for ss1, ss2 in zip(list(s1), list(wt)):
                if not ss1 == ss2:
                    ret += 1
            return ret


        def translate(s):
            try:
                return str(Seq(s).reverse_complement().translate(to_stop=True))
            except:
                return ""


        def substitutions(s):
            r = []
            for i, (ss, ww) in enumerate(zip(list(s), list(wt))):
                if ss != ww:
                    r.append("{}{}{}".format(ww, i + 1, ss))

            return ":".join(r)


        df = pd.read_hdf(input[0])
        df["cds"] = df.concensus_cds.apply(translate)
        df["substitutions"] = df.cds.apply(substitutions)

        df = df[df.cds.str.len() == 360]
        df = df[df["total_counts"] > 3000]
        df = df[df.hasConfidentCds]
        df["distance"] = df.cds.apply(hamming)
        df = df[df.distance <= 13]

        # extract logged sensor params
        convert = {
            "ic50": "ec50",
            "high-level": "ginf",
            "low-level": "g0",
        }
        for i, c in enumerate(["low-level", "high-level", "ic50"]):

            df[convert[c]] = df["log_{}".format(c.replace("-", "_"))]
            df["{}-std".format(convert[c])] = df["log_{} error".format(c.replace("-", "_"))]
            df["{}-var".format(convert[c])] = df["{}-std".format(convert[c])] ** 2

        # compute normalized values
        for i, c in enumerate(["ec50", "ginf", "g0"]):

            mu = df[c].mean()
            std = df[c].std()

            df[c + "-norm"] = (df[c] - mu) / std
            df[c + "-norm-std"] = df[c + "-std"] / std
            df[c + "-norm-var"] = df[c + "-norm-std"] ** 2

        # pull out desired columns for export
        filter_cols = (
            [
                "cds",
                "substitutions",
                "dual_BC_ID",
                "distance",
                "ec50",
                "ginf",
                "g0",
                "Low Level",
                "Low Level error",
                "High Level",
                "High Level error",
                "IC50",
                "IC50 error",
            ]
            + [f"{param}-std" for param in ["ec50", "ginf", "g0"]]
            + [f"{param}-var" for param in ["ec50", "ginf", "g0"]]
            + [f"{param}-norm" for param in ["ec50", "ginf", "g0"]]
            + [f"{param}-norm-std" for param in ["ec50", "ginf", "g0"]]
            + [f"{param}-norm-var" for param in ["ec50", "ginf", "g0"]]
        )

        for p in filter_cols[1:]:
            df = df[~(df[p].isnull())]

        # mutations in non-selection regions
        df = df[df["pacbio_KAN_mutations"] <= 0]
        df = df[df["pacbio_Ori_mutations"] <= 0]
        df = df[df["pacbio_tetA_mutations"] <= 0]
        df = df[df["pacbio_YFP_mutations"] <= 0]
        df = df[df["pacbio_insulator_mutations"] <= 0]

        df = df[df.sensor_GP_g_var.apply(lambda x: x.shape[0]) == 12]

        cv = KFold(10, shuffle=True, random_state=19033851)
        df["cv"] = np.nan
        for i, (_, cvts) in enumerate(cv.split(np.arange(len(df)))):
            df["cv"].iloc[cvts] = i

        df = df[df["log_high_level error"] < 0.7]
        df = df[df["good_hill_fit_points"] == 12]

        for i, (_, cvts) in enumerate(cv.split(np.arange(len(df)))):
            df["cv"].iloc[cvts] = i

        # add "dummy" variance to ignore variance on certain dimensions
        df["dummy-var"] = 0.0

        df[filter_cols + ["cv", "dummy-var"]].to_csv(output[0])
