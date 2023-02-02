from vaep.io.types import resolve_type
from snakemake.utils import min_version

min_version("6.0")


configfile: "config/config_grid.yaml"

# prefix: "grid_search" # could be used to redirect all outputs


# reuse rules from single experiment
# https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html#snakefiles-modules
module single_experiment:
    snakefile:
        "Snakefile"
    config:
        config


# could be in a subkey of the dictionary
GRID = {
    k: config[k]
    for k in [
        "epochs_max",
        "latent_dim",
        "hidden_layers",
    ]
}

name_template = config["name_template"]

folder_grid_search = config["folder_grid_search"]
folder_experiment = config["folder_experiment"]
folder_experiment2 = config[
    "folder_experiment2"
]  # expand fct, replaces single {} by double {{}}


AE_MODELS = ["DAE", "VAE"]
CF_MODEL = "CF"
MODELS = ["median", "interpolated", CF_MODEL, *AE_MODELS]


rule all:
    input:
        expand(
            f"{folder_experiment}/hyperpar_{{split}}_results_by_parameters_na_interpolated.pdf",
            level=config["levels"],
            split=["test_fake_na", "valid_fake_na"],
        ),
        f"{folder_grid_search}/average_performance_over_data_levels_best_test.pdf",


rule results:
    input:
        metrics=f"{folder_experiment}/all_metrics.csv",
        config=f"{folder_experiment}/all_configs.csv",
    output:
        expand(
            f"{folder_experiment2}/hyperpar_{{split}}_results_by_parameters_na_interpolated.pdf",
            split=["test_fake_na", "valid_fake_na"],
        ),
        f"{folder_experiment}/metrics_long_df.csv",
    params:
        models=MODELS,
    log:
        notebook=f"{folder_experiment}/02_3_grid_search_analysis.ipynb",
    notebook:
        "../02_3_grid_search_analysis.ipynb"


rule compare_search_by_dataset:
    input:
        expand(f"{folder_experiment}/metrics_long_df.csv", level=config["levels"]),
    output:
        f"{folder_grid_search}/average_performance_over_data_levels_best_test.pdf",
    log:
        notebook=f"{folder_grid_search}/best_models_over_all_data.ipynb",
    params:
        models=config["models"],
    notebook:
        "../02_4_best_models_over_all_data.ipynb"


nb = "01_0_split_data.ipynb"


use rule create_splits from single_experiment as splits with:
    input:
        nb=nb,
        configfile=config["config_split"],
    output:
        train_split=f"{folder_experiment}/data/train_X.pkl",
        nb=f"{folder_experiment}/{nb}",
    params:
        folder_experiment=f"{folder_experiment}",


# use rule train_models from single_experiment as train_AE_MODELS with:
rule train_ae_models:
    input:
        nb="01_1_train_{ae_model}.ipynb",
        train_split=f"{folder_experiment}/data/train_X.pkl",
        configfile=f"{folder_experiment}/"
        f"{name_template}/config_train_HL_{{hidden_layers}}.yaml",
    output:
        nb=f"{folder_experiment}/{name_template}/01_1_train_HL_{{hidden_layers}}_{{ae_model}}.ipynb",
        metric=f"{folder_experiment}/{name_template}/metrics/metrics_hl_{{hidden_layers}}_{{ae_model}}.json",
        config=f"{folder_experiment}/{name_template}/models/model_config_hl_{{hidden_layers}}_{{ae_model}}.yaml",
    params:
        folder_experiment=f"{folder_experiment}/{name_template}",
        model_key="HL_{hidden_layers}_{ae_model}",
    threads: 10
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"


use rule train_models from single_experiment as train_collab_model with:
    input:
        nb="01_1_train_{model}.ipynb",
        train_split=f"{folder_experiment}/data/train_X.pkl",
        configfile=f"{folder_experiment}/" f"{name_template}/config_train_collab.yaml",
    output:
        nb=f"{folder_experiment}/{name_template}/01_1_train_{{model}}.ipynb",
        metric=f"{folder_experiment}/{name_template}/metrics/metrics_{{model}}.json",
        config=f"{folder_experiment}/{name_template}/models/model_config_{{model}}.yaml",
    threads: 10
    params:
        folder_experiment=f"{folder_experiment}/{name_template}",


rule build_train_config:
    output:
        config_train=f"{folder_experiment}/"
        f"{name_template}/config_train_HL_{{hidden_layers}}.yaml",
    params:
        folder_data=f"{folder_experiment}/data/",
        batch_size=config["batch_size"],
        cuda=config["cuda"],
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}
        config["hidden_layers"] = wildcards["hidden_layers"]
        config["folder_experiment"] = str(PurePosixPath(output.config_train).parent)
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        config["batch_size"] = params.batch_size
        config["cuda"] = params.cuda
        with open(output.config_train, "w") as f:
            yaml.dump(config, f)


rule build_train_config_collab:
    output:
        config_train=f"{folder_experiment}/" f"{name_template}/config_train_collab.yaml",
    params:
        folder_data=f"{folder_experiment}/data/",
        batch_size_collab=config["batch_size_collab"],
        cuda=config["cuda"],
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}

        config["folder_experiment"] = str(PurePosixPath(output.config_train).parent)
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        config["batch_size"] = params.batch_size_collab
        config["cuda"] = params.cuda
        with open(output.config_train, "w") as f:
            yaml.dump(config, f)


rule collect_all_configs:
    input:
        expand(
            f"{folder_experiment2}/"
            f"{name_template}/models/model_config_hl_{{hidden_layers}}_{{ae_model}}.yaml",
            **GRID,
            ae_model=AE_MODELS,
        ),
        expand(
            f"{folder_experiment2}/{name_template}/models/model_config_{{collab_model}}.yaml",
            **GRID,
            collab_model=CF_MODEL,
        ),
    output:
        out=f"{folder_experiment}/all_configs.csv",
    log:
        notebook=f"{folder_experiment}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


rule collect_metrics:
    input:
        expand(
            f"{folder_experiment2}/{name_template}/metrics/metrics_hl_{{hidden_layers}}_{{ae_model}}.json",
            ae_model=AE_MODELS,
            **GRID,
        ),
        expand(
            f"{folder_experiment2}/{name_template}/metrics/metrics_{{collab_model}}.json",
            collab_model=CF_MODEL,
            **GRID,
        ),
    output:
        out=f"{folder_experiment}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"
