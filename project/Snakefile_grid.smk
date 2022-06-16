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

print(GRID)

name_template= config['name_template']
print(name_template)


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


use rule train_models from single_experiment  as model with:
    input:
        nb="14_experiment_03_train_{model}.ipynb",
        train_split="{folder_experiment}/data/train_X.pkl",
        configfile="{folder_experiment}/"
                   f"{name_template}/config_train.yaml"
    output:
        nb=f"{{folder_experiment}}/{name_template}/14_experiment_03_train_{{model}}.ipynb",
        metric=f"{{folder_experiment}}/{name_template}/metrics/metrics_{{model}}.json",
        config=f"{{folder_experiment}}/{name_template}/models/model_config_{{model}}.yaml"
    params:
        folder_experiment=f"{{folder_experiment}}/{name_template}"

# rule build_split_config:
#     output:
#         split="{folder_experiment}/config_split.yaml",

rule build_train_config:
    output:
        config_train="{folder_experiment}/"
             f"{name_template}/config_train.yaml"
    params:
        folder_data="{folder_experiment}/data/"
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


rule collect_all_configs:
    input:
        configs=expand("{folder_experiment}/"
                       f"{name_template}/models/model_config_{{model}}",
                folder_experiment=config["folder_experiment"],
                **GRID,
                model=['collab.yaml', 'dae.yaml', 'vae.yaml'])
    output:
        out = "{folder_experiment}/all_configs.json",
    log:
        notebook="{folder_experiment}/14_aggregate_configs.ipynb"
    notebook:
        "14_aggregate_configs.py.ipynb"
    # run:
    #     import json
    #     import yaml
    #     from pathlib import Path
    #     all = {}
    #     for fname in input:
    #         key = Path(fname).parent.name
    #         with open(fname) as f:
    #             all[key] = yaml.safe_load(f)
    #     with open(output.out, 'w') as f:
    #         json.dump(all, f)


rule collect_metrics:
    input:
        expand(f"{{folder_experiment}}/{name_template}/metrics/metrics_{{model}}.json",
            folder_experiment=config["folder_experiment"],
            model=['collab', 'dae', 'vae'],
            **GRID),
    output:
        out = "{folder_experiment}/all_metrics.json",
    run:
        import json
        from pathlib import Path
        import pandas as pd
        import vaep.pandas
        all_metrics = {}
        for fname in input:
            key = Path(fname).parents[1].name  # "grandparent" directory gives name
            with open(fname) as f:
                loaded = json.load(f)
            loaded = vaep.pandas.flatten_dict_of_dicts(loaded)
            if key not in all_metrics:
                all_metrics[key] = loaded
                continue
            for k, v in loaded.items():
                if k in all_metrics[key]:
                    assert all_metrics[key][k] == v, "Diverging values for {k}: {v1} vs {v2}".format(
                        k=k,
                        v1=all_metrics[key][k],
                        v2=v)
                else:
                    all_metrics[key][k] = v
        
        pd.DataFrame(all_metrics).to_json(output.out)