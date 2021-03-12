import os

subworkflow data:
    snakefile:
        "data.smk"

rule cv:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"]))
    output:
        os.path.join("experiments", config["name"], "{model}", "cv{cv}", "model.pt")
        # expand("experiments/{ds}", ds=config["name"]) + "/{model}/cv{cv}/model.pt"
    shell:
        "echo test"
