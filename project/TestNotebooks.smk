notebookes = [
    "1_data_exploration_peptides.ipynb",           
    "1_data_exploration_proteins.ipynb",           
    "01_FASTA_data_agg_by_gene.ipynb",             
    "01_FASTA_tryptic_digest.ipynb",               
    # "2_clustering_proteins.ipynb", # Reference Annelaura
    "02_data_available.ipynb",
    "02_data_exploration_peptides.ipynb",
    "02_data_exploration_proteins.ipynb",
    "02_metadata_rawfiles.ipynb",
    "02_summaries.ipynb",
    "04_all_raw_files.ipynb",                      
    "09_data_available.ipynb",                     
    "11_training_data_exploration_peptides.ipynb", 
    "11_select_data.ipynb",
    "12_experiment_01_fastai_version.ipynb",       
    "12_experiment_01_small_example.ipynb",        
    "12_experiment_01_transforms.ipynb",           
    "13_experiment_02_data.ipynb",                 
    "13_experiment_02_poster.ipynb",               
    "13_experiment_02.ipynb",                      
    "14_experiment_03_data_support.ipynb",         
    "14_experiment_03_data.ipynb",                 
    "14_experiment_03_dataloaders.ipynb",          
    "14_experiment_03_dataset.ipynb",              
    # "14_experiment_03_hyperpara_analysis.ipynb",   # needs parametrization for testing
    "14_experiment_03_latent_space_analysis.ipynb",
    "99_illustrations.ipynb",
    "embeddings.ipynb",
    "id_mapper.ipynb",
    "json_formats.ipynb",
    "sampling_in_pandas.ipynb",
    "VAEP_01_MaxQuantOutput.ipynb",
    # "VAEP_POC.ipynb", # to discard
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
