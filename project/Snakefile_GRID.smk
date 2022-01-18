import pathlib

GRID = {
    'epochs': [30],
    'latend_dim': [10, 25, 50, 75, 100],
    'hidden_layers': [1,2,3,4],
    'batch_size': [32, 64],
}

FOLDER = 'runs/experiment03'
pathlib.Path(FOLDER).mkdir(parents=True, exist_ok=True)

# import itertools 
# itertools.product(*GRID.values())

name_template = "experiment_HL_{hidden_layers}_LD_{latend_dim}_E_{epochs}_BS_{batch_size}"

rule all:
    input:
        expand(
        f"{{folder}}/{name_template}/{name_template}.md",
        folder=FOLDER,
        hidden_layers= GRID['hidden_layers'],
        latend_dim= GRID['latend_dim'],
        epochs= GRID["epochs"],
        batch_size = GRID['batch_size']
        )


rule execute_nb:
    input:
        notebook = "14_experiment_03_latent_space_analysis.ipynb"
    output:
         f"{{folder}}/{name_template}/{name_template}.ipynb"
    params:
        out_folder = f"{{folder}}/{name_template}"
    shell:
        "papermill {input.notebook} {output}"
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
