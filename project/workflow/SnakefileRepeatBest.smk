config_folder = 'config/repeat_best'

configfile: f'{config_folder}/config_repeat_best.yaml'


folder_experiment = config['folder'] + "/{level}"
folder_experiment2 = config['folder'] + "/{{level}}"

config['folder_experiment'] = folder_experiment

module single_experiment:
    snakefile: "Snakefile"
    config: config

rule all:
    input:
        f"{config['folder']}/model_performance_repeated_runs.pdf"


rule plot:
    input:
        configs=expand(
            f"{folder_experiment}/models/model_config_{{model}}_{{repeat}}.yaml",
            level=config['levels'],
            model=['DAE', 'VAE', 'collab'],
            repeat=range(config['repeats'])
        ),
        metrics=expand(
            f"{folder_experiment}/metrics/metrics_{{model}}_{{repeat}}.json",
            level=config['levels'],
            model=['DAE', 'VAE', 'collab'],
            repeat=range(config['repeats'])
        )
    output:
        f"{config['folder']}/model_performance_repeated_runs.pdf"
    log:
        notebook=f"{config['folder']}/14_best_models_repeated.ipynb"
    notebook:
        "14_best_models_repeated.ipynb"

nb='14_experiment_03_data.ipynb'
use rule create_splits from single_experiment as splits with:
    input:
        nb=nb,
        configfile=config['config_split']
    output:
        train_split=f"{folder_experiment}/data/train_X.pkl",
        nb=f"{folder_experiment}/{nb}",
    params:
        folder_experiment=f"{folder_experiment}"

# use rule train_models from single_experiment as train_models with:
rule train_models:
    input:
        nb="14_experiment_03_train_{model}.ipynb",
        train_split=f"{folder_experiment}/data/train_X.pkl",
        configfile=f"{config_folder}/{{model}}.yaml"
    output:
        nb=f"{folder_experiment}/14_experiment_03_train_{{model}}_{{repeat}}.ipynb",
        metric=f"{folder_experiment}/metrics/metrics_{{model}}_{{repeat}}.json",
        config=f"{folder_experiment}/models/model_config_{{model}}_{{repeat}}.yaml"
    params:
        folder_experiment=folder_experiment,
        model_key="{model}_{repeat}"
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r model_key {params.model_key}" # new in comparison to Sankefile
        " && jupyter nbconvert --to html {output.nb}"   
