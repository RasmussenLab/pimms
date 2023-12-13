from pathlib import Path
from vaep.io.types import resolve_type
from snakemake.utils import min_version
from snakemake.logging import logger

min_version("6.0")


configfile: "config/single_dev_dataset/proteinGroups_N50/config.yaml"


# prefix: "grid_search" # could be used to redirect all outputs


# reuse rules from single experiment
# https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html#snakefiles-modules
# config of this workflow can be passed to module. otherwise original module config is used
module single_experiment:
    snakefile:
        "Snakefile"
    config:
        config


root_experiment = Path(config["folder_experiment"])

# runs/dev_dataset_small/proteinGroups_N50
folder_experiment = config["folder_experiment"][:-2] + "{N}"
config["folder_experiment"] = folder_experiment


logger.info(f"{folder_experiment = }")
logger.info(f"{root_experiment = }")


rule all:
    input:
        combined_xlsx=f"{root_experiment.parent}/{root_experiment.name}_all_small.xlsx",


rule combine_result_tables:
    input:
        expand(
            [
                f"{folder_experiment}/01_2_performance_summary.xlsx",
            ],
            N=[
                10,
                20,
                30,
                40,
            ],
        ),
    output:
        combined_xlsx=f"{root_experiment.parent}/{root_experiment.name}_all_small.xlsx",
    notebook:
        "../03_3_combine_experiment_result_tables.ipynb"


# import after first rule
use rule * from single_experiment exclude create_splits, comparison, all as base_*


# non-rule python statements in original workflow
logger.info(f"{single_experiment.MODELS = }")
MODELS = single_experiment.MODELS

# logger.info(f"{config['models'] = }")
# logger.info(f"{config['NAGuideR_methods'] = }")


# # MODELS = config["models"].copy()
# # # ! needed to run NAGuideR methods, but needs to be switched off for comparison nb
# # # ? how is the original config imported here?
# # if config["NAGuideR_methods"]:
# #     MODELS += config["NAGuideR_methods"]


nb = "01_2_performance_plots.ipynb"


use rule comparison from single_experiment as adapted_comparison with:
    input:
        nb=nb,
        runs=expand(
            "{folder_experiment}/preds/pred_test_{model}.csv",
            folder_experiment="{folder_experiment}",
            model=MODELS,
            N="{N}",
        ),
    output:
        xlsx="{folder_experiment}/01_2_performance_summary.xlsx",
        pdf="{folder_experiment}/figures/2_1_test_errors_binned_by_int.pdf",
        nb="{folder_experiment}" f"/{nb}",


##########################################################################################
# Create Data splits
# separate workflow by level -> provide custom configs
nb = "01_0_split_data.ipynb"


rule create_splits:
    input:
        nb=nb,
        configfile=config["config_split"],
    output:
        train_split=f"{folder_experiment}/data/train_X.csv",
        nb=f"{folder_experiment}" f"/{nb}",
    params:
        folder_experiment=f"{folder_experiment}",
        meta_data=config["fn_rawfile_metadata"],
        sample_N=True,
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -p fn_rawfile_metadata {params.meta_data}"
        " -p select_N {wildcards.N}"
        " -p sample_N {params.sample_N}"
        " && jupyter nbconvert --to html {output.nb}"
