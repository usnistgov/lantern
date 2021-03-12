import os

subworkflow data:
    snakefile:
        "data.smk"

rule cv:
    input:
        data(expand("data/processed/{name}.csv", name=config["name"]))
    output:
        expand("experiments/{ds}/{model}/cv{cv}/model.pt", ds=config["name"], allow_missing=True)
    shell:
        "echo test"
