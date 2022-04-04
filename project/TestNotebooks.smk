notebookes = [
    "01_FASTA_data_agg_by_gene.ipynb",
    "01_FASTA_tryptic_digest.ipynb",
    # "2_clustering_proteins.ipynb", # Reference Annelaura
    "02_data_available.ipynb",
    "02_data_exploration_peptides.ipynb",
    "02_data_exploration_proteins.ipynb",
    "02_metadata_rawfiles.ipynb",
    "02_summaries.ipynb",
    "04_all_raw_files.ipynb",
    "11_select_data.ipynb",
    "11_training_data_exploration_peptides.ipynb",
    "14_experiment_03_data_support.ipynb",         
    "14_experiment_03_data.ipynb",                 
    "14_experiment_03_dataloaders.ipynb",          
    "14_experiment_03_dataset.ipynb",              
    # "14_experiment_03_hyperpara_analysis.ipynb",   # needs parametrization for testing
    "14_experiment_03_latent_space_analysis.ipynb",
    "15_embeddings.ipynb",
    "15_illustrations.ipynb",
    "15_pytorch_fastai_dataloaders.ipynb",
    "15_pytorch_fastai_dataset.ipynb",
    "erda_00_maxquant_file_reader.ipynb",
    "erda_01_mq_aggregate_summaries.ipynb",
    "erda_02_mq_count_peptides.ipynb",
    # "erda_10_training_data.ipynb",
    # "erda_11_select_training_data.ipynb",
    # "erda_12_explore_raw_MQ_data.ipynb",
    # "VAEP_POC.ipynb", # to discard
    # "id_mapper.ipynb", # to discard
]

rule run:
    input:
        expand("test_nb/{file}",
        file=notebookes)

rule execute:
    input:
        nb = "{file}",
    output:
        nb = "test_nb/{file}",
    # conda:
    #     vaep
    shell:
        "papermill {input.nb} {output.nb}"
