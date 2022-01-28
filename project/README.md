# Paper project

## Workflows

```
# cwd: project folder
snakemake --snakefile Snakefile_2D_latent.smk -p --jobs 1 -n
```


## Notebooks
for | notebook  | Description
--- | ---  |  --- 
erda | 00_mq_aggregate_summaries.ipynb   | Aggregate current summary files from MQ runs
erda | 00_mq_count_peptides.ipynb        | Aggregate peptide information from all MQ peptides.txt files <br> Saves processed file used for data selection (10_training_data)
_ | 01_explore_raw_MQ_data.ipynb      | Load an MQ txt output folder and browse data
_ | 01_FASTA_data_agg_by_gene.ipynb   | Investigate possibility to join proteins by gene
_ | 01_FASTA_tryptic_digest.ipynb     | Analyze fasta file used for peptide identification
_ | 04_all_raw_files.ipynb            | Find duplicate raw files, analyze sizes
_ | 1_data_exploration_peptides.ipynb | Describe current peptides training data
_ | 1_data_exploration_proteins.ipynb | \[NEEDS UPDATE\] Describe protein training data 
_ | 1_maxquant_file_analysing.ipynb   | \[NEEDS UPDATE\] how many peptides are available
_ | 09_data_available.ipynb           | Plots on theoretically available data
erda | 10_training_data.ipynb            | ERDA: Build training data dump
_ | 11_training_data_exploration_peptides.ipynb | Analyze dump of training data for patterns<br>  Visualize key metrics
_ | 12_experiment_01_small_example |  Assess inital model performance on smaller training data<br> - linear vs log transformed data<br> - vary number of layers and neurons in layers<br> - compare   | performance in original space
_ | 13_experiment_03*.ipynb           | See snakemake workflow
_ | 14_experiment_03_data_support.ipynb | Support of training data samples on selected training data
_ | 14_experiment_03_data.ipynb       | Create train, validation and test data splits
_ | 14_experiment_03_dataloaders.ipynb| Dataloading functionality
_ | 14_experiment_03_dataset.ipynb    | Dataset functionality
_ | 14_experiment_03_hyperpara_analysis.ipynb | Analyze different runs with varying hyperparameters on a dataset
_ | 14_experiment_03_latent_space_analysis.ipynb | Single run of all three models on a dataset
_ | 2_clustering_proteins.ipynb       | \[NEEDS UPDATE\] PCA protein analysis from Annelaura w/ initial   | data
 _ | 3_select_data.ipynb              | Visualize all data (and try running models) <br> 16GB RAM needed
_ | 3_select_training_data.ipynb      | \[NEEDS UPDATE\] Sort training data by gene
_ | 5_0_summaries                     | Analyzse summaries.txt data from all samples
_ | id_mapper.ipynb                   | train models per gene, see overlaps in proteins, see coverage   | of proteins with observed peptides, align overlapping peptide sequences
_ | json_formats.ipynb                | Investigate storring training data as json with correct encoding
_ | VAEP_01_MaxQuantOutput.ipynb      | Show MaxQuantOutput class behaviour
_ | VAEP_POC.ipynb                    | Presentation for POC

> exectued notebook versions are stored under [`doc/ipynbs`](doc/ipynbs) exported as markdown docs


## Notebook descriptions


### `1_data_exploration_peptides.ipynb` 


### 