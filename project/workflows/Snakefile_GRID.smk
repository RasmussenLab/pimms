import pathlib


configfile: "config/config_grid.yaml"
configfile: "config/config_paths.yaml"

GRID = {k:config[k] 
        for k 
        in ['epochs',
            'latent_dim',
            'hidden_layers',
            'batch_size']
        }

FOLDER = config['FOLDER']
pathlib.Path(FOLDER).mkdir(parents=True, exist_ok=True)

name_template = config['name_template']
DATA = config['DATA']


rule all:
    input:
        expand(
        f"{{folder}}/{name_template}{{files}}",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latent_dim= GRID['latent_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size'],
        files = ['/metrics.json']#, f'/{name_template}.md']
        ),
        expand(
        f"{{folder}}/{name_template}/{name_template}.md",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latent_dim= GRID['latent_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size']
        ),
        f"{FOLDER}/analysis_metrics.ipynb",

rule analyze_metrics:
    input:
        metrics = "{folder}/all_metrics.json",
        configs = "{folder}/all_configs.json",
        nb = "14_experiment_03_hyperpara_analysis.ipynb"
    output:
        "{folder}/analysis_metrics.ipynb"
    shell:
        "papermill {input.nb} {output}"
        " -p metrics_json {input.metrics}"
        " -p configs_json {input.configs}"

rule collect_metrics:
    input:
        expand(
        f"{{folder}}/{name_template}/metrics.json",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latent_dim= GRID['latent_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size']
        )
    output:
        out = "{folder}/all_metrics.json",
    run:
        import json
        from pathlib import Path
        all_metrics = {}
        for fname in input:
            key = Path(fname).parent.name
            with open(fname) as f:
                all_metrics[key] = json.load(f)
        with open(output.out, 'w') as f:
            json.dump(all_metrics, f)


rule collect_all_configs:
    input:
        expand(
        f"{{folder}}/{name_template}/model_config.yml",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latent_dim= GRID['latent_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size']
        )
    output:
        out = "{folder}/all_configs.json",
    run:
        import json
        import yaml
        from pathlib import Path
        all = {}
        for fname in input:
            key = Path(fname).parent.name
            with open(fname) as f:
                all[key] = yaml.safe_load(f)
        with open(output.out, 'w') as f:
            json.dump(all, f)


rule execute_nb:
    input:
        expand("{{folder}}/data/{split}",
        split=['train_X', 'val_X', 'test_X', 'val_y', 'test_y', 'freq_train.csv']),
        nb = "14_experiment_03_latent_space_analysis.ipynb",
    output:
        nb = f"{{folder}}/{name_template}/{name_template}.ipynb",
        metrics = f"{{folder}}/{name_template}/metrics.json",
        config = f"{{folder}}/{name_template}/model_config.yml",
    params:
        out_folder = f"{{folder}}/{name_template}",
        data = "{folder}/data",
    shell:
        "papermill {input.nb} {output.nb}"
        " -p data {params.data}"
        " -p latent_dim {wildcards.latent_dim}"
        " -p hidden_layers {wildcards.hidden_layers}"
        " -p n_epochs {wildcards.epochs}"
        " -p out_folder {params.out_folder}"
        " -p batch_size {wildcards.batch_size}"
        " -p force_train True"


rule covert_to_md:
    input:
        f"{{folder}}/{name_template}/{name_template}.ipynb"
    output:
        f"{{folder}}/{name_template}/{name_template}.md"
    run:
        commands = [
            'jupyter nbconvert --to markdown {input}',
            'nbconvert_md_processing -v -i {output} --overwrite'
        ]
        for c in commands:
            shell(c)


rule data_splits:
    input:
        file = DATA,
        nb = "14_experiment_03_data.ipynb"
    output:
        expand("{{folder}}/data/{split}",
        split=['train_X', 'val_X', 'test_X', 'val_y', 'test_y', 'freq_train.csv']),
        nb="{folder}/data_selection.ipynb",
    params:
        query = config['QUERY_SUBSET']
    shell:
        "papermill {input.nb} {output.nb}"
        ' -p query_subset_meta  "{params.query}"'
        " -p FN_PEPTIDE_INTENSITIES {input.file}"
        " -p experiment_folder {wildcards.folder}"