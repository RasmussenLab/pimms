notebookes = [
    "misc_embeddings.ipynb",
    "misc_illustrations.ipynb",
    "misc_pytorch_fastai_dataloaders.ipynb",
    "misc_pytorch_fastai_dataset.ipynb",
    "misc_sampling_in_pandas.ipynb"
]


rule run:
    input:
        expand("test_nb/{file}", file=notebookes),


rule execute:
    input:
        nb="{file}",
    output:
        nb="test_nb/{file}",
    # conda:
    #     vaep
    shell:
        "papermill {input.nb} {output.nb}"
