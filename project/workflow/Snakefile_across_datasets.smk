config_folder = 'config/across_datasets'

configfile: f'{config_folder}/config.yaml'


folder_experiment = config['folder'] + "/{level}"
# folder_experiment = config['folder'] + "/{level}" + "/{dataset}"
folder_experiment2 = config['folder'] + "/{{level}}"

config['folder_experiment'] = folder_experiment


rule all:
    input:
        f"{config['folder']}/model_performance_repeated_runs.pdf"


rule plot:
    input:
        configs=expand(
            f"{folder_experiment}/{{dataset}}/models/model_config_{{model}}.yaml",
            level=config['levels'],
            model=['DAE', 'VAE', 'collab'],
            dataset=config['datasets']
        ),
        metrics=expand(
            f"{folder_experiment}/{{dataset}}/metrics/metrics_{{model}}.json",
            level=config['levels'],
            model=['DAE', 'VAE', 'collab'],
            dataset=config['datasets']
        ),
    output:
        f"{config['folder']}/model_performance_repeated_runs.pdf"
    log:
        notebook=f"{config['folder']}/14_best_models_repeated.ipynb"
    notebook:
        "../14_across_datasets.ipynb"


def get_fn_intensities(wildcards):
    """Some metadata is stored in folder name which leads to the need for a lookup of names"""
    ret = f'data/single_datasets/{config["data_folders"][wildcards.level]}/{wildcards.dataset}.pkl'
    return ret


nb='14_experiment_03_data.ipynb'
rule create_splits:
    input:
        nb=nb,
        configfile=f'{config_folder}/{config["config_split"]}'
    output:
        train_split=f"{folder_experiment}/{{dataset}}/data/train_X.pkl",
        nb=f"{folder_experiment}/{{dataset}}/{nb}",
    params:
        folder_experiment=f"{folder_experiment}/{{dataset}}",
        FN_INTENSITIES= get_fn_intensities
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r FN_INTENSITIES {params.FN_INTENSITIES}"
        " && jupyter nbconvert --to html {output.nb}"
    

rule train_models:
    input:
        nb="14_experiment_03_train_{model}.ipynb",
        train_split=f"{folder_experiment}/{{dataset}}/data/train_X.pkl",
        configfile=f"{config_folder}/{{model}}.yaml"
    output:
        nb=f"{folder_experiment}/{{dataset}}/14_experiment_03_train_{{model}}.ipynb",
        metric=f"{folder_experiment}/{{dataset}}/metrics/metrics_{{model}}.json",
        config=f"{folder_experiment}/{{dataset}}/models/model_config_{{model}}.yaml"
    params:
        folder_experiment=f"{folder_experiment}/{{dataset}}",
        model_key="{model}"
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r model_key {params.model_key}" # new in comparison to Sankefile
        " && jupyter nbconvert --to html {output.nb}"   
