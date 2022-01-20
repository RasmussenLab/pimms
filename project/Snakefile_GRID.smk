import pathlib


configfile: "config_grid.yaml"
configfile: "config_paths.yaml"

GRID = {k:config[k] 
        for k 
        in ['epochs',
            'latend_dim',
            'hidden_layers',
            'batch_size']
        }

FOLDER = config['FOLDER']
pathlib.Path(FOLDER).mkdir(parents=True, exist_ok=True)

name_template = config['name_template']


rule all:
    input:
        expand(
        f"{{folder}}/{name_template}/metrics.json",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latend_dim= GRID['latend_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size']
        ),
        f"{FOLDER}/all_metrics.json",


rule collect_metrics:
    input:
        expand(
        f"{{folder}}/{name_template}/metrics.json",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latend_dim= GRID['latend_dim'],
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


rule execute_nb:
    input:
        nb = "14_experiment_03_latent_space_analysis.ipynb"
    output:
        nb = f"{{folder}}/{name_template}/{name_template}.ipynb",
        metrics = f"{{folder}}/{name_template}/metrics.json"
    params:
        out_folder = f"{{folder}}/{name_template}"
    shell:
        "papermill {input.nb} {output.nb}"
        " -p latend_dim {wildcards.latend_dim}"
        " -p hidden_layers {wildcards.hidden_layers}"
        " -p n_epochs {wildcards.epochs}"
        " -p out_folder {params.out_folder}"
        " -p batch_size {wildcards.batch_size}"


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
