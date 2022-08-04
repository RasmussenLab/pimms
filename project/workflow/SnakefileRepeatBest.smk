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
        metrics=f"{config['folder']}/metrics.pkl"
    output:
        f"{config['folder']}/model_performance_repeated_runs.pdf"
    params:
        repitition_name=config['repitition_name']
    log:
        notebook=f"{config['folder']}/14_best_models_repeated.ipynb"
    notebook:
        "../14_best_models_repeated.ipynb"

rule collect_metrics:
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
        f"{config['folder']}/metrics.pkl"
    params:
        folder=config['folder'],
        repitition_name=config['repitition_name']
    run:
        from pathlib import Path
        import pandas as pd
        import vaep.models

        REPITITION_NAME=params.repitition_name

        def select_content(s:str):
            s = s.split('metrics_')[1]
            assert isinstance(s, str), f"More than one split: {s}"
            model, repeat = s.split('_')
            return model, int(repeat)
            
        # test_cases = ['model_metrics_DAE_0',
        #             'model_metrics_VAE_3',
        #             'model_metrics_collab_2']
        
        # for test_case in test_cases:
        #     print(f"{test_case} = {select_content(test_case)}")

        def key_from_fname(fname:Path):
            model, repeat = select_content(fname.stem)
            key = (fname.parents[1].name, repeat)
            return key

        all_metrics = vaep.models.collect_metrics(input.metrics, key_from_fname)
        metrics = pd.DataFrame(all_metrics).T
        metrics.index.names = ('data level', REPITITION_NAME)
        metrics

        FOLDER = Path(params.folder)

        metrics = metrics.T.sort_index().loc[pd.IndexSlice[['NA interpolated', 'NA not interpolated'],
                                                ['valid_fake_na', 'test_fake_na'],
                                                ['median', 'interpolated', 'collab', 'DAE', 'VAE'],
                                                :]]
        metrics.to_csv(FOLDER/ "metrics.csv")
        metrics.to_excel(FOLDER/ "metrics.xlsx")
        metrics.to_pickle(FOLDER/ "metrics.pkl")

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
