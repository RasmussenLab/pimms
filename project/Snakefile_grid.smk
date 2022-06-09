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
            'hidden_layers',
            'batch_size']
        }

print(GRID)

name_template= config['name_template']
print(name_template)

rule all:
    input:
        expand(f"{{folder_experiment}}/{name_template}/{{nb}}",
            folder_experiment=config["folder_experiment"],
            nb=['14_experiment_03_train_collab.ipynb',
                '14_experiment_03_train_dae.ipynb',
                '14_experiment_03_train_vae.ipynb'
        ],
       **GRID)

use rule create_splits from single_experiment as splits with:
    input:
        nb='14_experiment_03_data.ipynb',
        configfile ="config/proteinGroups_split.yaml"


use rule train_models from single_experiment  as model with:
    input:
        nb="{nb}",
        train_split="{folder_experiment}/data/train_X.pkl",
        configfile="{folder_experiment}/"
                   f"{name_template}/config_train.yaml"
    output:
        nb=f"{{folder_experiment}}/{name_template}/{{nb}}",
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

        config = {k: resolve_type(v) for k, v in wildcards.items()}
        config['folder_experiment'] = str(PurePosixPath(output.config_train).parent) 
        config['folder_data'] = params.folder_data
        with open(output.config_train, 'w') as f:
            yaml.dump(config, f)


