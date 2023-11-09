config_folder = "config/across_datasets"


configfile: f"{config_folder}/config.yaml"


folder_experiment = config["folder"] + "/{level}"
# folder_experiment = config['folder'] + "/{level}" + "/{dataset}" # Possibility
folder_experiment2 = config["folder"] + "/{{level}}"

config["folder_experiment"] = folder_experiment


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
            f"{folder_experiment}/{{dataset}}/models/model_config_{{model}}.yaml",
            level=config["levels"],
            model=["DAE", "VAE", "collab"],
            dataset=config["datasets"],
        ),
        metrics=expand(
            f"{folder_experiment}/{{dataset}}/metrics/metrics_{{model}}.json",
            level=config["levels"],
            model=["DAE", "VAE", "collab"],
            dataset=config["datasets"],
        ),
    output:
        f"{config['folder']}/metrics.pkl",
    params:
        folder=config["folder"],
        repitition_name=config["repitition_name"],
    run:
        from pathlib import Path
        import pandas as pd
        import vaep.models

        REPITITION_NAME = params.repitition_name


        # key fully specified in path
        def key_from_fname(fname):
            key = (fname.parents[2].name, fname.parents[1].name)
            return key


        all_metrics = vaep.models.collect_metrics(input.metrics, key_from_fname)
        metrics = pd.DataFrame(all_metrics).T
        metrics.index.names = ("data level", REPITITION_NAME)
        metrics

        FOLDER = Path(params.folder)

        metrics = metrics.T.sort_index().loc[
            pd.IndexSlice[
                ["NA interpolated", "NA not interpolated"],
                ["valid_fake_na", "test_fake_na"],
                ["median", "interpolated", "collab", "DAE", "VAE"],
                :,
            ]
        ]
        metrics.to_csv(FOLDER / "metrics.csv")
        metrics.to_excel(FOLDER / "metrics.xlsx")
        metrics.to_pickle(FOLDER / "metrics.pkl")


def get_fn_intensities(wildcards):
    """Some metadata is stored in folder name which leads to the need for a lookup of names"""
    ret = f'data/dev_datasets/{config["data_folders"][wildcards.level]}/{wildcards.dataset}.pkl'
    return ret


nb = "01_0_split_data.ipynb"


rule create_splits:
    input:
        nb=nb,
        configfile=f'{config_folder}/{config["config_split"]}',
    output:
        train_split=f"{folder_experiment}/{{dataset}}/data/train_X.pkl",
        nb=f"{folder_experiment}/{{dataset}}/{nb}",
    params:
        folder_experiment=f"{folder_experiment}/{{dataset}}",
        FN_INTENSITIES=get_fn_intensities,
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r FN_INTENSITIES {params.FN_INTENSITIES}"
        " && jupyter nbconvert --to html {output.nb}"


rule train_models:
    input:
        nb="01_1_train_{model}.ipynb",
        train_split=f"{folder_experiment}/{{dataset}}/data/train_X.pkl",
        configfile=f"{config_folder}/{{model}}.yaml",
    output:
        nb=f"{folder_experiment}/{{dataset}}/01_1_train_{{model}}.ipynb",
        metric=f"{folder_experiment}/{{dataset}}/metrics/metrics_{{model}}.json",
        config=f"{folder_experiment}/{{dataset}}/models/model_config_{{model}}.yaml",
    params:
        folder_experiment=f"{folder_experiment}/{{dataset}}",
        model_key="{model}",
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"
        # new in comparison to Sankefile
