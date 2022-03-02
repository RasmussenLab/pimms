# Paper project

## Workflows

```
# cwd: project folder
snakemake --snakefile Snakefile_2D_latent.smk -p --jobs 1 -n
```


## Notebooks
for | notebook  | Description
--- | ---  |  --- 
erda | 00_mq_aggregate_summaries.ipynb | Aggregate current summary files from MQ runs
erda | 00_mq_count_peptides.ipynb      | Aggregate peptide information from all MQ peptides.txt files <br> Saves processed file used for data selection (10_training_data)
erda | 01_explore_raw_MQ_data.ipynb    | Load an MQ txt output folder and browse data
erda | 10_training_data.ipynb          | ERDA: Build training data dump
erda | 3_select_training_data.ipynb    | \[NEEDS UPDATE\] Sort training data by gene
_ | 01_FASTA_data_agg_by_gene.ipynb    | Investigate possibility to join proteins by gene
_ | 01_FASTA_tryptic_digest.ipynb      | Analyze fasta file used for peptide identification
_ | 02_data_exploration_peptides.ipynb | Describe current peptides training data
_ | 02_data_exploration_proteins.ipynb | \[NEEDS UPDATE\] Describe protein training data 
_ | 02_summaries.ipynb                 | Analyzse summaries.txt data from all samples
_ | 04_all_raw_files.ipynb             | Find duplicate raw files, analyze sizes
_ | 02_data_available.ipynb            | Plots on theoretically available data
_ | 11_training_data_exploration_peptides.ipynb | Analyze dump of training data for patterns<br>  Visualize key metrics
_ | 12_experiment_01_small_example     |  Assess inital model performance on smaller training data<br> - linear vs log transformed data<br> - vary number of layers and neurons in layers<br> - compare   | performance in original space
_ | 13_experiment_03*.ipynb            | See snakemake workflow
_ | 14_experiment_03_data_support.ipynb| Support of training data samples on selected training data
_ | 14_experiment_03_data.ipynb        | Create train, validation and test data splits
_ | 14_experiment_03_dataloaders.ipynb | Dataloading functionality
_ | 14_experiment_03_dataset.ipynb     | Dataset functionality
_ | 14_experiment_03_hyperpara_analysis.ipynb | Analyze different runs with varying hyperparameters on a dataset
_ | 14_experiment_03_latent_space_analysis.ipynb | Single run of all three models on a dataset
_ | 2_clustering_proteins.ipynb        | \[documentation\] PCA protein analysis from Annelaura w/ initial data <br> (Executed, only for documentation)
_ | 3_select_data.ipynb                | Visualize all data (and try running models) <br> 16GB RAM needed
_ | id_mapper.ipynb                    | train models per gene, see overlaps in proteins, see coverage   | of proteins with observed peptides, align overlapping peptide sequences
_ | json_formats.ipynb                 | Investigate storring training data as json with correct encoding
_ | VAEP_01_MaxQuantOutput.ipynb       | Show MaxQuantOutput class behaviour
_ | VAEP_POC.ipynb                     | Presentation for POC
_ | embeddings.ipynb                   | Fastai Embedding class
_ | id_mapper.ipynb                    | explore peptides?
_ | json_formats.ipynb                 | json formats with a schema: example and limitation
_ | run_ipynbs.py                      | Example script of for running notebooks on cmd (w/o papermill)
_ | sampling_in_pandas.ipynb           | How to sample in pandas
_ | VAEP_01_MaxQuantOutput.ipynb       | Analyze MQ output
_ | VAEP_POC.ipynb                     | First POC analysis

> exectued notebook versions are stored under [`doc/ipynbs`](doc/ipynbs) exported as markdown docs


# Notebook descriptions

## erda notebooks

- [ ] determine order and rename accordingly with prefix
## 01 Analysis Fasta

### `01_FASTA_tryptic_digest.ipynb`

- analysis FASTA file used for protein search

### `01_FASTA_data_agg_by_gene.ipynb`

- analysis of gene to protein mapping of fasta file

## 02 Analysis dataset

### `02_data_available.ipynb`
- analyze `count_all_peptides.json`: How many peptides are identified overall in all
  processed files 

> erda notebook: `00_mq_count_peptides.ipynb`

### `02_data_exploration_peptides.ipynb` 
- finds files originationg from fractionation experiments
- plot mask indicating presence/abscence of peptide measurement in an experiment
- intensity log-transformation: 

### `02_data_exploration_proteins.ipynb` 


### `02_summaries.ipynb`
- analysze all `summaries.txt`

### `02_clustering_proteins.ipynb`

- total amount of peptide data
## Training data
### `11_select_data.ipynb`

### `11_training_data_exploration_peptides.ipynb`

## Latest Model training

### `12_experiment_01_fastai_version.ipynb`

### `12_experiment_01_small_example.ipynb`

### `12_experiment_01_transforms.ipynb`

### `13_experiment_02_data.ipynb`

### `13_experiment_02_poster.ipynb`

### `13_experiment_02.ipynb`

### `14_experiment_03_data_support.ipynb`

### `14_experiment_03_data.ipynb`
### `14_experiment_03_dataloaders.ipynb`

### `14_experiment_03_dataset.ipynb`
### `14_experiment_03_hyperpara_analysis.ipynb`

### `14_experiment_03_latent_space_analysis.ipynb`

## Misc
### `99_illustrations.ipynb`
- illustrations for presentations
- e.g. shifted normal imputation

### `embeddings.ipynb`
### `id_mapper.ipynb`

### `json_formats.ipynb`

### `run_ipynbs.py`

### `sampling_in_pandas.ipynb`

### `VAEP_01_MaxQuantOutput.ipynb`

### `VAEP_POC.ipynb`