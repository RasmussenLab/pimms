from vaep.io.types import resolve_type
from snakemake.utils import min_version

min_version("6.0")


configfile: "config/grid_search_large_data/config_grid_small.yaml"


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

folder_grid_search = config["folder_grid_search"]
folder_dataset = f"{folder_grid_search}/{{level}}"  # rename to folder_dataset
folder_dataset2 = (
    f"{folder_grid_search}" "/{{level}}"
)  # expand fct, replaces single {} by double {{}}
root_model = f"{folder_dataset}/models/{{model}}"

print(folder_dataset)
print(folder_dataset2)

MODELS = config["models"]


wildcard_constraints:
    level="|".join(config["levels"]),
    ae_model="|".join(["DAE", "VAE"]),


rule all:
    input:
        expand(
            f"{folder_dataset}/hyperpar_{{split}}_results_by_parameters.pdf",
            level=config["levels"],
            split=["test_fake_na", "valid_fake_na"],
        ),
        f"{folder_grid_search}/average_performance_over_data_levels_best_test.pdf",


##########################################################################################
# per model per dataset -> one metrics_long_df.csv # decide on format
rule compare_all:
    input:
        expand(f"{folder_dataset}/metrics_long_df.csv", level=config["levels"]),
    output:
        f"{folder_grid_search}/average_performance_over_data_levels_best_test.pdf",
    log:
        notebook=f"{folder_grid_search}/best_models_over_all_data.ipynb",
    params:
        models=MODELS,
    notebook:
        "../02_4_best_models_over_all_data.ipynb"


# split once for each dataset
nb = "01_0_split_data.ipynb"


use rule create_splits from single_experiment as splits with:
    input:
        nb=nb,
        configfile=config["config_split"],
    output:
        train_split=f"{folder_dataset}/data/train_X.csv",
        nb=f"{folder_dataset}/{nb}",
    params:
        folder_experiment=f"{folder_dataset}",
        meta_data=config["fn_rawfile_metadata"],


##########################################################################################
rule results_dataset:
    input:
        metrics=f"{folder_dataset}/all_metrics.csv",
        config=f"{folder_dataset}/all_configs.csv",
    output:
        expand(
            f"{folder_dataset2}/hyperpar_{{split}}_results_by_parameters.pdf",
            split=["test_fake_na", "valid_fake_na"],
        ),
        f"{folder_dataset}/metrics_long_df.csv",
    params:
        models=MODELS,
        file_format=config["file_format"],
    log:
        notebook=f"{folder_dataset}/02_3_grid_search_analysis.ipynb",
    notebook:
        "../02_3_grid_search_analysis.ipynb"


rule collect_all_metrics:
    input:
        expand(
            f"{folder_dataset2}/models/{{model}}/all_metrics.csv",
            model=MODELS,
        ),
    output:
        out=f"{folder_dataset}/all_metrics.csv",
    log:
        notebook=f"{folder_dataset}/02_1_join_metrics.ipynb",
    notebook:
        "../02_1_join_metrics.py.ipynb"


rule collect_all_configs:
    input:
        expand(
            f"{folder_dataset2}/models/{{model}}/all_configs.csv",
            model=MODELS,
        ),
    output:
        f"{folder_dataset}/all_configs.csv",
    log:
        notebook=f"{folder_dataset}/02_2_join_configs.ipynb",
    notebook:
        "../02_2_join_configs.py.ipynb"


##########################################################################################

### AE based models
_model = "VAE"
root_model = f"{folder_dataset}/models/{_model}"
root_model2 = f"{folder_dataset2}/models/{_model}"
run_id_template = "LD_{latent_dim}_E_{epochs_max}_HL_{hidden_layers}"

print(GRID)


rule collect_VAE_configs:
    input:
        expand(
            f"{root_model2}/{run_id_template}/model_config_{{model}}.yaml",
            **GRID,
            model=_model,
        ),
    output:
        out=f"{root_model}/all_configs.csv",
    log:
        notebook=f"{root_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


rule collect_metrics_vae:
    input:
        expand(
            f"{root_model2}/{run_id_template}/metrics_{{model}}.json",
            model=_model,
            **GRID,
        ),
    output:
        out=f"{root_model}/all_metrics.csv",
    log:
        notebook=f"{root_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"


_model = "DAE"
root_model = f"{folder_dataset}/models/{_model}"
root_model2 = f"{folder_dataset2}/models/{_model}"
run_id_template = "LD_{latent_dim}_E_{epochs_max}_HL_{hidden_layers}"


rule collect_DAE_configs:
    input:
        expand(
            f"{root_model2}/{run_id_template}/model_config_{{model}}.yaml",
            **GRID,
            model=_model,
        ),
    output:
        out=f"{root_model}/all_configs.csv",
    log:
        notebook=f"{root_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


rule collect_metrics_dae:
    input:
        expand(
            f"{root_model2}/{run_id_template}/metrics_{{model}}.json",
            model=_model,
            **GRID,
        ),
    output:
        out=f"{root_model}/all_metrics.csv",
    log:
        notebook=f"{root_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"


root_model = f"{folder_dataset}/models/{{model}}"
root_model2 = f"{folder_dataset2}" "/models/{{model}}"
run_id_template = "LD_{latent_dim}_E_{epochs_max}_HL_{hidden_layers}"


rule train_ae_models:
    input:
        nb="01_1_train_{ae_model}.ipynb",
        train_split=f"{folder_dataset}/data/train_X.csv",
        configfile=f"{root_model}/{run_id_template}/config_nb_train.yaml",
    output:
        nb=f"{root_model}/{run_id_template}/01_1_train_{{ae_model}}.ipynb",
        metric=f"{root_model}/{run_id_template}/metrics_{{ae_model}}.json",
        config=f"{root_model}/{run_id_template}/model_config_{{ae_model}}.yaml",
    params:
        folder_dataset=f"{root_model}/{run_id_template}",
        # model_key="HL_{hidden_layers}_LD_{hidden_layers}",  # ToDo
    # add log
    # https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#log-files
    threads: 10
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_dataset}"
        " && jupyter nbconvert --to html {output.nb}"



rule build_train_config_ae:
    output:
        config_train=f"{root_model}/{run_id_template}/config_nb_train.yaml",
    params:
        folder_data=f"{folder_dataset}/data/",
        batch_size=config["batch_size"],
        cuda=config["cuda"],
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}
        config["hidden_layers"] = wildcards["hidden_layers"]
        # config["folder_experiment"] = str(PurePosixPath(output.config_train).parent)
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        config["batch_size"] = params.batch_size
        config["cuda"] = params.cuda
        with open(output.config_train, "w") as f:
            yaml.dump(config, f)


### Collaborative Filtering (CF)
_model = "CF"
root_model = f"{folder_dataset}/models/{_model}"
root_model2 = f"{folder_dataset2}/models/{_model}"
run_id_template = "LD_{latent_dim}_E_{epochs_max}"


rule collect_CF_configs:
    input:
        expand(
            f"{root_model2}/{run_id_template}/model_config_{{model}}.yaml",
            **GRID,
            model=_model,
        ),
    output:
        out=f"{root_model}/all_configs.csv",
    log:
        notebook=f"{root_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


rule collect_metrics_cf:
    input:
        expand(
            f"{root_model2}/{run_id_template}/metrics_{{model}}.json",
            **GRID,
            model=_model,
        ),
    output:
        out=f"{root_model}/all_metrics.csv",
    log:
        notebook=f"{root_model}/02_1_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"


rule train_CF_model:
    input:
        nb=f"01_1_train_{_model}.ipynb",
        train_split=f"{folder_dataset}/data/train_X.csv",
        configfile=f"{root_model}/" f"{run_id_template}/config_nb_train_CF.yaml",
    output:
        nb=f"{root_model}/{run_id_template}/01_1_train_{_model}.ipynb",
        metric=f"{root_model}/{run_id_template}/metrics_{_model}.json",
        config=f"{root_model}/{run_id_template}/model_config_{_model}.yaml",
    benchmark:
        f"{root_model}/{run_id_template}/01_1_train_{_model}.tsv"
    threads: 10
    params:
        folder_experiment=f"{root_model}/{run_id_template}",
        meta_data=config["fn_rawfile_metadata"],
        model_key=f"{_model}",
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -p fn_rawfile_metadata {params.meta_data}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"


rule build_train_config_collab:
    output:
        config_nb_train=f"{root_model}/{run_id_template}/config_nb_train_CF.yaml",
    params:
        folder_data=f"{folder_dataset}/data/",
        # folder_experiment=root_model,
        batch_size_collab=config["batch_size_collab"],
        cuda=config["cuda"],
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}

        # config["folder_experiment"] = params.root_model
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        config["batch_size"] = params.batch_size_collab
        config["cuda"] = params.cuda
        with open(output.config_nb_train, "w") as f:
            yaml.dump(config, f)


### Median imputation
_model = "Median"
root_model = f"{folder_dataset}/models/{_model}"


rule collect_metrics_median:
    input:
        f"{root_model}/metrics_{_model}.json",
    output:
        out=f"{root_model}/all_metrics.csv",
    log:
        notebook=f"{root_model}/02_2_aggregate_metrics.ipynb",
    notebook:
        "../02_1_aggregate_metrics.py.ipynb"


rule collect_median_configs:
    input:
        f"{root_model}/model_config_{_model}.yaml",
    output:
        out=f"{root_model}/all_configs.csv",
    log:
        notebook=f"{root_model}/02_2_aggregate_configs.ipynb",
    notebook:
        "../02_2_aggregate_configs.py.ipynb"


rule build_train_config_median:
    output:
        config_train=f"{root_model}/config_nb_{_model}.yaml",
    params:
        folder_data=f"{folder_dataset}/data/",
        fn_rawfile_metadata=config["fn_rawfile_metadata"],
        # folder_experiment=root_model,
    run:
        from pathlib import PurePosixPath
        import yaml

        config = dict(wildcards)  # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != "hidden_layers"}

        # config["folder_experiment"] = params.root_model
        config["fn_rawfile_metadata"] = params.fn_rawfile_metadata
        config["folder_data"] = params.folder_data
        with open(output.config_train, "w") as f:
            yaml.dump(config, f)


rule use_median_model:
    input:
        nb="01_1_train_{model}.ipynb",
        train_split=f"{folder_dataset}/data/train_X.csv",
        config_train=f"{root_model}/config_nb_{{model}}.yaml",
    output:
        nb=f"{root_model}/01_1_train_{{model}}.ipynb",
        metric=f"{root_model}/metrics_{{model}}.json",
        config=f"{root_model}/model_config_{{model}}.yaml",
    params:
        folder_dataset=f"{root_model}",
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.config_train}"
        " -r folder_experiment {params.folder_dataset}"
        " && jupyter nbconvert --to html {output.nb}"
