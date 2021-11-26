import pathlib

GRID = {
    'epochs': [10, 30],
    'feat': [25, 50, 75, 100, 200, 300, 400, 500, 1000]
}

FOLDER = 'runs/2D'
# pathlib.Path(FOLDER).mkdir(parents=True, exist_ok=True)

# import itertools 
# itertools.product(*GRID.values())


rule all:
    input:
        expand("{folder}/latent_2D_{feat}_{epochs}.md", folder=FOLDER, feat= GRID['feat'], epochs= GRID["epochs"])
                     

rule execute_nb:
    input:
        notebook = "13_experiment_02_poster.ipynb"
    output:
        "{folder}/latent_2D_{feat}_{epochs}.ipynb"
    shell:
        "papermill {input.notebook} {output} -p n_feat {wildcards.feat} -p n_epochs {wildcards.epochs} -p out_folder {wildcards.folder}"
        
rule covert_to_md:
    input:
        notebook = "{folder}/latent_2D_{feat}_{epochs}.ipynb"
    output:
        converted = "{folder}/latent_2D_{feat}_{epochs}.md"
    # shell:
    #     "jupyter nbconvert --to markdown {input.notebook} && nbconvert_md_processing -v -i {output.converted} --overwrite"
    run:
        commands = [
            'jupyter nbconvert --to markdown {input.notebook}',
            'nbconvert_md_processing -v -i {output.converted} --overwrite'
        ]
        for c in commands:
            shell(c)