config_folder = "config/repeat_best"


configfile: f"{config_folder}/config_repeat_best.yaml"


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
    run:
        from pathlib import Path
        import pandas as pd
        import vaep.models.collect_dumps

        REPITITION_NAME = params.repitition_name


        def select_content(s: str):
            s = s.split("metrics_")[1]
            assert isinstance(s, str), f"More than one split: {s}"
            model, repeat = s.split("_")
            return model, int(repeat)

        def key_from_fname(fname: Path):
            _, repeat = select_content(fname.stem)
            key = (fname.parent.name, repeat)
            return key

        def load_metric_file(fname: Path, frist_split='metrics'):
            fname = Path(fname)
            with open(fname) as f:
                loaded = json.load(f)
            loaded = vaep.pandas.flatten_dict_of_dicts(loaded)
            key = key_from_fname(fname) # '_'.join(key_from_fname(fname))
            return key, loaded


        all_metrics = vaep.models.collect_dumps.collect(input.metrics, load_metric_file)
        metrics = pd.DataFrame(all_metrics)
        metrics = metrics.set_index('id')
        metrics.index = pd.MultiIndex.from_tuples(
                                    metrics.index,
                                    names=("data level", REPITITION_NAME))
        metrics.columns = pd.MultiIndex.from_tuples(
                                        metrics.columns, 
                                        names=('data_split', 'model', 'metric_name'))
        metrics = (metrics
                   .stack(['metric_name', 'model'])
                   .unstack(['model', 'metric_name'])
                   .T)

        FOLDER = Path(params.folder)
        metrics = metrics.loc[
            pd.IndexSlice[
                ["valid_fake_na", "test_fake_na"],
                ["CF", "DAE", "VAE"],
                :]
        ]
        metrics.to_csv(FOLDER / "metrics.csv")
        metrics.to_excel(FOLDER / "metrics.xlsx")
        metrics.to_pickle(FOLDER / "metrics.pkl")



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
    shell:
        "papermill {input.nb} {output.nb}"
        " -f {input.configfile}"
        " -r folder_experiment {params.folder_experiment}"
        " -r fn_rawfile_metadata {params.meta_data}"
        " -r file_format {params.file_format}"
        " -r model_key {params.model_key}"
        " && jupyter nbconvert --to html {output.nb}"
