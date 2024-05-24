"""
Document how all the notebooks for a single experiment are connected.
"""
from snakemake.logging import logger


configfile: "config/single_dev_dataset/proteinGroups_N50/config_v2.yaml"


MAX_WALLTIME = "24:00:00"
# Thinnode resources sharing: 40 cores and 196 GB RAM (minus 2GB for snakemake)
# JOB_RAM_MB = int(204_800 / 40 * config['THREATS_MQ'])
JOB_RAM_MB = "8gb"
folder_experiment = config["folder_experiment"]
logger.info(f"{folder_experiment = }")
logger.info(f"{config = }")


# local rules are excuted in the process (job) running snakemake
localrules:
    all,
    comparison,
    transform_NAGuideR_predictions,
    transform_data_to_wide_format,
    create_splits,
    dump_train_config,
    dump_split_config,


rule all:
    input:
        f"{folder_experiment}/figures/2_1_test_errors_binned_by_int.pdf",
        f"{folder_experiment}/01_2_performance_summary.xlsx",


nb = "01_2_performance_plots.ipynb"

if "frac_mnar" in config:
    config["split_data"]["frac_mnar"] = config["frac_mnar"]

# print(config['split_data'])
# MODELS = config["models"].copy()

MODELS = list()
model_configs = dict()
for m in config["models"]:
    for model, cfg_model in m.items():
        MODELS.append(model)
        model_configs[model] = dict(cfg_model)
else:
    del model, cfg_model

if config["NAGuideR_methods"]:
    MODELS += config["NAGuideR_methods"]

nb_stem = "01_2_performance_summary"


rule comparison:
    input:
        nb=nb,
        runs=expand(
            "{folder_experiment}/preds/pred_test_{model}.csv",
            folder_experiment=config["folder_experiment"],
            model=MODELS,
        ),
    output:
        xlsx=f"{{folder_experiment}}/{nb_stem}.xlsx",
        pdf="{folder_experiment}/figures/2_1_test_errors_binned_by_int.pdf",
        nb="{folder_experiment}" f"/{nb}",
    params:
        meta_data=config["fn_rawfile_metadata"],
        models=",".join(MODELS),
        err=f"{{folder_experiment}}/{nb_stem}.e",
        out=f"{{folder_experiment}}/{nb_stem}.o",
    shell:
        "papermill {input.nb} {output.nb}"
        " -r fn_rawfile_metadata {params.meta_data:q}"
        " -r folder_experiment {wildcards.folder_experiment:q}"
        " -r models {params.models:q}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# train NaGuideR methods
nb_stem = "01_1_transfer_NAGuideR_pred"


rule transform_NAGuideR_predictions:
    input:
        dumps=expand(
            "{{folder_experiment}}/preds/pred_all_{method}.csv",
            method=config["NAGuideR_methods"],
        ),
        nb=f"{nb_stem}.ipynb",
    output:
        # "{{folder_experiment}}/preds/pred_real_na_{method}.csv"),
        expand(
            (
                "{{folder_experiment}}/preds/pred_val_{method}.csv",
                "{{folder_experiment}}/preds/pred_test_{method}.csv",
            ),
            method=config["NAGuideR_methods"],
        ),
        nb="{folder_experiment}/01_1_transfer_NAGuideR_pred.ipynb",
    benchmark:
        "{folder_experiment}/" f"{nb_stem}.tsv"
    params:
        err=f"{{folder_experiment}}/{nb_stem}.e",
        out=f"{{folder_experiment}}/{nb_stem}.o",
        folder_experiment="{folder_experiment}",
        # https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#non-file-parameters-for-rules
        dumps_as_str=lambda wildcards, input: ",".join(input.dumps),
    shell:
        "papermill {input.nb} {output.nb}"
        " -r folder_experiment {params.folder_experiment}"
        " -p dumps {params.dumps_as_str}"
        " && jupyter nbconvert --to html {output.nb}"


rule train_NAGuideR_model:
    input:
        nb="01_1_train_NAGuideR_methods.ipynb",
        train_split="{folder_experiment}/data/data_wide_sample_cols.csv",
    output:
        nb="{folder_experiment}/01_1_train_NAGuideR_{method}.ipynb",
        dump="{folder_experiment}/preds/pred_all_{method}.csv",
    resources:
        mem_mb=JOB_RAM_MB,
        walltime=MAX_WALLTIME,
    threads: 1  # R is single threaded
    benchmark:
        "{folder_experiment}/01_1_train_NAGuideR_{method}.tsv"
    params:
        err="{folder_experiment}/01_1_train_NAGuideR_{method}.e",
        out="{folder_experiment}/01_1_train_NAGuideR_{method}.o",
        folder_experiment="{folder_experiment}",
        method="{method}",
        name="{method}",
    conda:
        "vaep"
    shell:
        "papermill {input.nb} {output.nb}"
        " -r train_split {input.train_split}"
        " -r method {params.method}"
        " -r folder_experiment {params.folder_experiment}"
        " && jupyter nbconvert --to html {output.nb}"


nb_stem = "01_0_transform_data_to_wide_format"


rule transform_data_to_wide_format:
    input:
        nb=f"{nb_stem}.ipynb",
        train_split="{folder_experiment}/data/train_X.csv",
    output:
        nb="{folder_experiment}/01_0_transform_data_to_wide_format.ipynb",
        train_split="{folder_experiment}/data/data_wide_sample_cols.csv",
    params:
        folder_experiment="{folder_experiment}",
        err=f"{{folder_experiment}}/{nb_stem}.e",
        out=f"{{folder_experiment}}/{nb_stem}.o",
    shell:
        "papermill {input.nb} {output.nb}"
        " -r folder_experiment {params.folder_experiment}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# train models in python
rule train_models:
    input:
        nb=lambda wildcards: "01_1_train_{}.ipynb".format(
            model_configs[wildcards.model]["model"]
        ),
        train_split="{folder_experiment}/data/train_X.csv",
        configfile=config["config_train"],
    output:
        nb="{folder_experiment}/01_1_train_{model}.ipynb",
        pred="{folder_experiment}/preds/pred_test_{model}.csv",
    benchmark:
        "{folder_experiment}/01_1_train_{model}.tsv"
    resources:
        mem_mb=JOB_RAM_MB,
        walltime=MAX_WALLTIME,
    params:
        folder_experiment="{folder_experiment}",
        meta_data=config["fn_rawfile_metadata"],
        err="{folder_experiment}/01_1_train_{model}.e",
        out="{folder_experiment}/01_1_train_{model}.o",
        name="{model}",
    conda:
        "vaep"
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -p fn_rawfile_metadata {params.meta_data}"
        " -r model_key {wildcards.model}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# create config file dumps for each model


rule dump_train_config:
    output:
        configfile=config["config_train"],
    run:
        import yaml

        with open(output.configfile, "w") as f:
            f.write("# Build in Snakemake workflow\n")
            yaml.dump(model_configs[wildcards.model], f, sort_keys=False)

##########################################################################################
# Create data splits
nb_stem = "01_0_split_data"

rule create_splits:
    input:
        nb=f"{nb_stem}.ipynb",
        configfile=config["config_split"],
    output:
        train_split="{folder_experiment}/data/train_X.csv",
        nb="{folder_experiment}" f"/{nb_stem}.ipynb",
    params:
        folder_experiment="{folder_experiment}",
        meta_data=config["fn_rawfile_metadata"],
        err=f"{{folder_experiment}}/{nb_stem}.e",
        out=f"{{folder_experiment}}/{nb_stem}.o",
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -p fn_rawfile_metadata {params.meta_data}"
        " && jupyter nbconvert --to html {output.nb}"


##########################################################################################
# create data splitting configuration file


rule dump_split_config:
    output:
        configfile=config["config_split"],
    run:
        import yaml
        # recreating dict, otherwise Null becomes string "Null" in yaml dump...
        cfg = dict()
        for k, v in config["split_data"].items():
            cfg[k] = v
        with open(output.configfile, "w") as f:
            f.write("# Build in Snakemake workflow (from v2)\n")
            yaml.dump(cfg, f, sort_keys=False)
