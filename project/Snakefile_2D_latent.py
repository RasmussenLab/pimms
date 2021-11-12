import pathlib

GRID = {
    'epochs': [10, 30],
    'feat': [25, 50, 75, 100, 200, 300, 400, 500, 1000]
}

FOLDER = 'runs/2D'
# Path(FOLDER).mkdir(parents=True, exist_ok=True)

# import itertools 
# itertools.product(*GRID.values())


rule all:
    input:
        expand("{folder}/latent_2D_{feat}_{epochs}.ipynb", folder=FOLDER, feat= GRID['feat'], epochs= GRID["epochs"])
        

rule execute_nb:
    input:
        notebook = "13_experiment_02_poster.ipynb"
    output:
        "{folder}/latent_2D_{feat}_{epochs}.ipynb"
    shell:
        "papermill {input.notebook} {output} -p n_feat {wildcards.feat} -p n_epochs {wildcards.epochs}" 
