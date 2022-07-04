from vaep.io.types import resolve_type
from snakemake.utils import min_version
min_version("6.0")

configfile: "config/config_grid.yaml"

# prefix: "grid_search" # could be used to redirect all outputs

# reuse rules from single experiment
# https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html#snakefiles-modules
module single_experiment:
    snakefile: "Snakefile"
    config: config

# could be in a subkey of the dictionary
GRID = {k:config[k] 
        for k 
        in ['epochs',
            'latent_dim',
            'hidden_layers', # collab does not change based on #hidden layers -> repeated computation
            ]
        }

# print(GRID)

name_template= config['name_template']
# print(name_template)


rule all:
    input:
        expand("{folder_experiment}/hyperpar_{split}_results.pdf",
        folder_experiment=config["folder_experiment"],
        split=["test_fake_na", "valid_fake_na"])

rule results:
    input:     
        metrics="{folder_experiment}/all_metrics.json",
        config="{folder_experiment}/all_configs.json"
    output:
        expand("{{folder_experiment}}/hyperpar_{split}_results.pdf",
            split=["test_fake_na", "valid_fake_na"])
    log:
        notebook="{folder_experiment}/14_experiment_03_hyperpara_analysis.ipynb"
    notebook:
        "14_experiment_03_hyperpara_analysis.ipynb"


use rule create_splits from single_experiment as splits with:
    input:
        nb='14_experiment_03_data.ipynb',
        configfile =config['config_split']



# use rule train_models from single_experiment  as model with:
#     input:
#         nb="14_experiment_03_train_{model}.ipynb",
#         train_split="{folder_experiment}/data/train_X.pkl",
#         configfile="{folder_experiment}/"
#                    f"{name_template}/config_train.yaml"
#     output:
#         nb=f"{{folder_experiment}}/{name_template}/14_experiment_03_train_{{model}}.ipynb",
#         metric=f"{{folder_experiment}}/{name_template}/metrics/metrics_{{model}}.json",
#         config=f"{{folder_experiment}}/{name_template}/models/model_config_{{model}}.yaml"
#     params:
#         folder_experiment=f"{{folder_experiment}}/{name_template}"


# use rule train_models from single_experiment as train_ae_models with:
rule train_ae_models:
    input:
        nb="14_experiment_03_train_{ae_model}.ipynb",
        train_split="{folder_experiment}/data/train_X.pkl",
        configfile="{folder_experiment}/"
                   f"{name_template}/config_train_HL_{{hidden_layers}}.yaml"
    output:
        nb=f"{{folder_experiment}}/{name_template}/14_experiment_03_train_HL_{{hidden_layers}}_{{ae_model}}.ipynb",
        metric=f"{{folder_experiment}}/{name_template}/metrics/metrics_HL_{{hidden_layers}}_{{ae_model}}.json",
        config=f"{{folder_experiment}}/{name_template}/models/model_config_HL_{{hidden_layers}}_{{ae_model}}.yaml"
    params:
        folder_experiment=f"{{folder_experiment}}/{name_template}",
        model_key="HL_{hidden_layers}_{ae_model}"
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"   

use rule train_models from single_experiment as train_collab_model with:
    input:
        nb="14_experiment_03_train_{collab_model}.ipynb",
        train_split="{folder_experiment}/data/train_X.pkl",
        configfile="{folder_experiment}/"
                   f"{name_template}/config_train_collab.yaml"
    output:
        nb=f"{{folder_experiment}}/{name_template}/14_experiment_03_train_{{collab_model}}.ipynb",
        metric=f"{{folder_experiment}}/{name_template}/metrics/metrics_{{collab_model}}.json",
        config=f"{{folder_experiment}}/{name_template}/models/model_config_{{collab_model}}.yaml"
    params:
        folder_experiment=f"{{folder_experiment}}/{name_template}"


rule build_train_config:
    output:
        config_train="{folder_experiment}/"
             f"{name_template}/config_train_HL_{{hidden_layers}}.yaml"
    params:
        folder_data="{folder_experiment}/data/",
    run:
        from pathlib import PurePosixPath
        import yaml
        config = dict(wildcards) # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != 'hidden_layers'}
        if '_' not in wildcards['hidden_layers']:
            config['hidden_layers'] = int(wildcards['hidden_layers'])
        else:
            config['hidden_layers'] = wildcards['hidden_layers']
        config['folder_experiment'] = str(PurePosixPath(output.config_train).parent) 
        config['folder_data'] = params.folder_data
        with open(output.config_train, 'w') as f:
            yaml.dump(config, f)

rule build_train_config_collab:
    output:
        config_train="{folder_experiment}/"
            f"{name_template}/config_train_collab.yaml"
    params:
        folder_data="{folder_experiment}/data/"
    run:
        from pathlib import PurePosixPath
        import yaml
        config = dict(wildcards) # copy dict
        config = {k: resolve_type(v) for k, v in config.items() if k != 'hidden_layers'}
        # if '_' not in wildcards['hidden_layers']:
        #     config['hidden_layers'] = int(wildcards['hidden_layers'])
        # else:
        #     config['hidden_layers'] = wildcards['hidden_layers']
        config['folder_experiment'] = str(PurePosixPath(output.config_train).parent) 
        config['folder_data'] = params.folder_data
        with open(output.config_train, 'w') as f:
            yaml.dump(config, f)


rule collect_all_configs:
    input:
        expand("{folder_experiment}/"
              f"{name_template}/models/model_config_HL_{{hidden_layers}}_{{ae_model}}.yaml",
                folder_experiment=config["folder_experiment"],
                **GRID,
                ae_model=['dae', 'vae']),
        expand(f"{{folder_experiment}}/{name_template}/models/model_config_{{collab_model}}.yaml",
                folder_experiment=config["folder_experiment"],
                **GRID,
                collab_model='collab')
    output:
        out = "{folder_experiment}/all_configs.json",
    log:
        notebook="{folder_experiment}/14_aggregate_configs.ipynb"
    notebook:
        "14_aggregate_configs.py.ipynb"


rule collect_metrics:
    input:
        expand(f"{{folder_experiment}}/{name_template}/metrics/metrics_HL_{{hidden_layers}}_{{ae_model}}.json",
            folder_experiment=config["folder_experiment"],
            ae_model=['dae', 'vae'],
            **GRID),
        expand(f"{{folder_experiment}}/{name_template}/metrics/metrics_{{collab_model}}.json",
                  folder_experiment=config["folder_experiment"],
                  collab_model=['collab'],
                  **GRID)
    output:
        out = "{folder_experiment}/all_metrics.json",
    log:
        notebook="{folder_experiment}/14_collect_all_metrics.ipynb"
    notebook:
        "14_collect_all_metrics.py.ipynb"