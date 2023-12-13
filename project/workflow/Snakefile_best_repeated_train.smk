config_folder = "config/repeat_best"


configfile: f"{config_folder}/train.yaml"


folder_experiment = config["folder"] + "/{level}"

config["folder_experiment"] = folder_experiment

MODELS = ["DAE", "VAE", "CF"]


rule all:
    input:
        f"{config['folder']}/model_performance_repeated_runs.pdf",


rule plot:
    input:
        metrics=f"{config['folder']}/metrics.pkl",
    output:
        f"{config['folder']}/model_performance_repeated_runs.pdf",
    params:
        repitition_name=config["repitition_name"],
    log:
        notebook=f"{config['folder']}/03_1_best_models_comparison.ipynb",
    notebook:
        "../03_1_best_models_comparison.ipynb"


rule collect_metrics:
    input:
        configs=expand(
            f"{folder_experiment}/model_config_{{model}}_{{repeat}}.yaml",
            level=config["levels"],
            model=MODELS,
            repeat=range(config["repeats"]),
        ),
        metrics=expand(
            f"{folder_experiment}/metrics_{{model}}_{{repeat}}.json",
            level=config["levels"],
            model=MODELS,
            repeat=range(config["repeats"]),
        ),
    output:
        f"{config['folder']}/metrics.pkl",
    params:
        folder=config["folder"],
        repitition_name=config["repitition_name"],
    log:
        notebook=f"{config['folder']}/collect_metrics.ipynb",
    notebook:
        "notebooks/best_repeated_train_collect_metrics.ipynb"


nb = "01_0_split_data.ipynb"


rule create_splits:
    input:
        nb=nb,
        configfile=config["config_split"],
    output:
        train_split=f"{folder_experiment}/data/train_X.{config['file_format']}",
        nb=f"{folder_experiment}/{nb}",
    params:
        folder_experiment=f"{folder_experiment}",
        meta_data=config["fn_rawfile_metadata"],
        file_format=config["file_format"],
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r fn_rawfile_metadata {params.meta_data}"
        " -r file_format {params.file_format}"
        " && jupyter nbconvert --to html {output.nb}"


rule train_models:
    input:
        nb="01_1_train_{model}.ipynb",
        train_split=f"{folder_experiment}/data/train_X.{config['file_format']}",
        configfile="config/single_dev_dataset/{level}/train_{model}.yaml",
    output:
        nb=f"{folder_experiment}/01_1_train_{{model}}_{{repeat}}.ipynb",
        metric=f"{folder_experiment}/metrics_{{model}}_{{repeat}}.json",
        config=f"{folder_experiment}/model_config_{{model}}_{{repeat}}.yaml",
    params:
        folder_experiment=f"{folder_experiment}",
        model_key="{model}_{repeat}",
        meta_data=config["fn_rawfile_metadata"],
        file_format=config["file_format"],
        cuda=config['cuda'],
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r fn_rawfile_metadata {params.meta_data}"
        " -r file_format {params.file_format}"
        " -r model_key {params.model_key}"
        " -p cuda {params.cuda}"
        " && jupyter nbconvert --to html {output.nb}"
