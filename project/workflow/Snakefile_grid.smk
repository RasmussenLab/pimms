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
# MODELS = ["median", "interpolated", CF_MODEL, *AE_MODELS]
MODELS = config['models']

wildcard_constraints:
    level="|".join(config["levels"])

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

# per model per dataset -> one metrics_long_df.csv # decide on format
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
        configfile=f"{folder_experiment}/" f"{name_template}/config_train_CF.yaml",
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
        config_train=f"{folder_experiment}/" f"{name_template}/config_train_CF.yaml",
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


rule collect_VAE_configs:
    input:
        expand(
            f"{folder_experiment2}/"
            f"{name_template}/models/model_config_hl_{{hidden_layers}}_{{model}}.yaml",
            **GRID,
            model='VAE',
        ),
    output:
        out=f"{folder_experiment}/{'VAE'}/all_configs.csv",
    log:
        notebook=f"{folder_experiment}/{'VAE'}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"

rule collect_DAE_configs:
    input:
        expand(
            f"{folder_experiment2}/"
            f"{name_template}/models/model_config_hl_{{hidden_layers}}_{{model}}.yaml",
            **GRID,
            model='DAE',
        ),
    output:
        out=f"{folder_experiment}/{'DAE'}/all_configs.csv",
    log:
        notebook=f"{folder_experiment}/{'DAE'}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"

### Median imputation
_model = 'median'

rule build_train_config_median:
    output:
        config_train=f"{folder_experiment}/models/model_config_{_model}.yaml",
    params:
        folder_data=f"{folder_experiment}/data/",
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}

        config["folder_experiment"] = str(PurePosixPath(output.config_train).parent)
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        with open(output.config_train, "w") as f:
            yaml.dump(config, f)


rule median_model:
    input:
        nb="01_1_train_{model}.ipynb",
        train_split=f"{folder_experiment}/data/train_X.pkl",
        configfile=f"{folder_experiment}/models/model_config_{{model}}.yaml",
    output:
        nb=f"{folder_experiment}/01_1_train_{{model}}.ipynb",
        metric=f"{folder_experiment}/metrics/metrics_{{model}}.json",
        config=f"{folder_experiment}/models/model_config_{{model}}.yaml",
    threads: 10
    params:
        folder_experiment=f"{folder_experiment}",
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " && jupyter nbconvert --to html {output.nb}"


rule collect_median_configs:
    input:
        f"{folder_experiment}/models/model_config_{_model}.yaml",
    output:
        out=f"{folder_experiment}/{_model}/all_configs.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


### 

_model = 'CF'
rule collect_CF_configs:
    input:
        expand(
            f"{folder_experiment2}/"
            f"{name_template}/models/model_config_hl_{{hidden_layers}}_{{model}}.yaml",
            **GRID,
            model=_model,
        ),
    output:
        out=f"{folder_experiment}/{_model}/all_configs.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"

rule collect_all_configs:
    input:
        expand(
            f"{folder_experiment2}/{{model}}/all_configs.csv",
            # levels=config["levels"],
            model=MODELS,
        ),
    output:
        f"{folder_experiment}/all_configs.csv"
    log:
        notebook=f"{folder_experiment}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


_model = 'VAE'
rule collect_metrics_vae:
    input:
        expand(
            f"{folder_experiment2}/{name_template}/metrics/metrics_hl_{{hidden_layers}}_{{model}}.json",
            model=_model,
            **GRID,
        ),
    output:
        out=f"{folder_experiment}/{_model}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"

_model = 'DAE'
rule collect_metrics_dae:
    input:
        expand(
            f"{folder_experiment2}/{name_template}/metrics/metrics_hl_{{hidden_layers}}_{{model}}.json",
            model=_model,
            **GRID,
        ),
    output:
        out=f"{folder_experiment}/{_model}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"

_model = 'CF'
rule collect_metrics_cf:
    input:
        expand(
            f"{folder_experiment2}/{name_template}/metrics/metrics_{{model}}.json",
            model=_model,
            **GRID,
        ),
    output:
        out=f"{folder_experiment}/{_model}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"

_model = 'median'
rule collect_metrics_median:
    input:
        f"{folder_experiment}/metrics/metrics_{_model}.json",
    output:
        out=f"{folder_experiment}/{_model}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/{_model}/02_2_aggregate_metrics.ipynb",
    notebook:
        "../02_2_aggregate_metrics.py.ipynb"


rule collect_all_metrics:
    input:
        expand(
            f"{folder_experiment2}/{{model}}/all_metrics.csv",
            model=MODELS,
        ),
    output:
        out=f"{folder_experiment}/all_metrics.csv",
    log:
        notebook=f"{folder_experiment}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"
