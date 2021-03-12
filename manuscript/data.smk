
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
