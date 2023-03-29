notebookes = [
    "misc_FASTA_data_agg_by_gene.ipynb",
    "misc_FASTA_tryptic_digest.ipynb",
    # "2_clustering_proteins.ipynb", # Reference Annelaura
    "erda_data_available.ipynb",
    "misc_data_exploration_peptides.ipynb",
    "misc_data_exploration_proteins.ipynb",
    "00_0_hela_metadata_rawfiles.ipynb",
    "00_1_hela_MQ_summaries.ipynb",
    "00_2_hela_all_raw_files.ipynb",
    "misc_protein_support.ipynb",
    "00_5_training_data_exploration.ipynb",
    "00_4_development_dataset_support.ipynb",
    "01_0_split_data.ipynb",
    "14_experiment_03_dataloaders.ipynb",
    "14_experiment_03_dataset.ipynb",
    # "02_3_grid_search_analysis.ipynb",   # needs parametrization for testing
    "14_experiment_03_latent_space_analysis.ipynb",
    "misc_embeddings.ipynb",
    "misc_illustrations.ipynb",
    "misc_pytorch_fastai_dataloaders.ipynb",
    "misc_pytorch_fastai_dataset.ipynb",
    "erda_00_maxquant_file_reader.ipynb",
    "erda_01_mq_aggregate_summaries.ipynb",
    "erda_02_mq_count_peptides.ipynb",
    # "erda_10_training_data.ipynb",
    # "erda_11_select_training_data.ipynb",
    # "erda_12_explore_raw_MQ_data.ipynb",
    # "VAEP_POC.ipynb", # to discard
    # "misc_id_mapper.ipynb", # to discard
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
