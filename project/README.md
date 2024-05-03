# Paper project
The PIMMS comparison workflow is a snakemake workflow that runs the all selected PIMMS models and R-models on 
a user-provided dataset and compares the results. An example for the smaller HeLa development dataset on the 
protein groups level is re-built regularly and available at: [rasmussenlab.org/pimms](https://www.rasmussenlab.org/pimms/)

## Data requirements

Required is abundance data in wide or long format in order to run the models. 

| Sample ID | Protein A | Protein B | Protein C | ... |
| --- | --- | --- | --- | --- |
| sample_01 | 0.1       | 0.2       | 0.3       | ... |
| sample_02 | 0.2       | 0.1       | 0.4       | ... |
| sample_03 | 0.3       | 0.2       | 0.1       | ... |

or as long formated data.

| Sample ID | Protein | Abundance |
| --- | --- | --- |
| sample_01 | Protein A | 0.1       |
| sample_01 | Protein B | 0.2       |
| sample_01 | Protein C | 0.3       |
| sample_02 | Protein A | 0.2       |
| sample_02 | Protein B | 0.1       |
| sample_02 | Protein C | 0.4       |
| sample_03 | Protein A | 0.3       |
| sample_03 | Protein B | 0.2       |
| sample_03 | Protein C | 0.1       |

Currently `pickle`d and `csv` files are supported.

Optionally, ThermoRawFileParser output cab be used as metadata.
along further as e.g. clinical metadata for each sample.
- `meta_date_col`: optional column used to order the samples (e.g. by date)
- `meta_cat_col`: optional categoyr column used for visualization of samples in PCAs

## Workflows

> `Snakefile` stored in [workflow](workflow/README.md) folder ([link](https://github.com/RasmussenLab/pimms/blob/HEAD/project/workflow))

Execute example data (50 runs of HeLa lysate on protein groups level):

```
# in ./project
snakemake -c1 -p -n # remove -n to execute
```

The example workflow runs in 3-5 mins on default setup (no GPU, no paralleziation).

### Setup development data

Setup project workflow
```
# see what is all executed
snakemake --snakefile Snakemake_project.smk -p -n # dry-run
``` 

### single experiments

Single Experiment with config files

```bash
# cwd: project folder (this folder)
snakemake --configfile config/single_dev_dataset/aggpeptides/config.yaml -p -n 
```

### Single notebooks using papermill

execute single notebooks
```bash
set DATASET=df_intensities_proteinGroups_long/Q_Exactive_HF_X_Orbitrap_6070 
papermill  01_0_split_data.ipynb --help-notebook # check parameters
papermill  01_0_split_data.ipynb runs/experiment_03/%DATASET%/experiment_03_data.ipynb -p MIN_SAMPLE 0.5 -p fn_rawfile_metadata data/dev_datasets/%DATASET%.csv -p index_col "Sample ID" -p columns_name peptide
```

## Notebooks

- run: a single experiment with models attached, see `workflow/Snakefile`
- grid: only grid search associated, see `workflow/Snakefile_grid.smk`
- best: best models repeatedly trained or across datasets, see `workflow/Snakefile_best_repeated_train.smk` and `workflow/Snakefile_best_across_datasets.smk`
- ald: ALD study associated, see `workflow/Sankefile_ald_comparison.smk`

tag | notebook  | Description
--- | ---  |  --- 
Single experiment |
run  | 01_0_split_data.ipynb               | Create train, validation and test data splits
run  | 01_1_train_<model>.ipynb            | Train a single model e.g. (VAE, DAE, CF)
run  | 01_2_performance_plots.ipynb        | Performance of single model run
Grid search and best model analysis |
grid | 02_1_aggregate_metrics.py.ipynb    | Aggregate metrics
grid | 02_2_aggregate_configs.py.ipynb    | Aggregate model configurations
grid | 02_3_grid_search_analysis.ipynb    | Analyze different runs with varying hyperparameters on a dataset
grid | 02_4_best_models_over_all_data     | Show best models and best models across data types
best | 03_1_best_models_comparison.ipynb  | best model trained repeatedly or across datasets
Applications |
ald | 16_ald_data.ipynb               | preprocess data -> could be move to data folder
ald | 16_ald_diff_analysis.ipynb      | differential analysis (DA), dump scores 
ald | 16_ald_compare_methods.ipynb    | DA comparison between methods
ald | 16_ald_ml_new_feat.ipynb        | ML model comparison
ald | 16_ald_compare_single_pg.ipynb  | [DEV] Compare imputation for feat between methods (dist plots)
Miscancellous notebooks on different topics (partly exploration) |
misc | misc_embeddings.ipynb                | FastAI Embeddings
misc | misc_illustrations.ipynb             | Illustrations of certain concepts (e.g. draw from shifted random distribution)
misc | misc_json_formats.ipynb              | Investigate storring training data as json with correct encoding
misc | misc_MaxQuantOutput.ipynb            | \[documentation\] Analyze MQ output, show MaxQuantOutput class behaviour
misc | misc_protein_support.ipynb           | peptide sequences mapped to protein sequences
misc | misc_pytorch_fastai_dataset.ipynb    | Dataset functionality
misc | misc_pytorch_fastai_dataloaders.ipynb| Dataloading functionality
misc | misc_sampling_in_pandas.ipynb        | How to sample in pandas

# Notebook descriptions (To be completed)

## Inspect dataset

### `00_5_training_data_exploration.py`

Can be execute manually

```bash
jupytext 00_5_training_data_exploration.py --to ipynb -o - | papermill - runs/example/00_5_training_data_exploration.ipynb -f config/single_dev_dataset/example/inspect_data.yaml
```

## Single experiment run
### `01_0_split_data.ipynb`

- select data according to procedure described in **Fig. S1**

### `01_1_train_<model>.ipynb`
- notebooks for training model `X` (e.g. `VAE`, `DAE` or `CF`)

### `01_2_performance_plots.ipynb`

## Grid search and best model analysis

### `02_1_aggregate_metrics.py.ipynb` and `02_1_join_metrics.py.ipynb`
- helper script to collect `metrics`. 
### `02_2_aggregate_configs.py.ipynb` and `02_2_join_configs.py.ipynb`

- helper script to collect `config`urations.

### `02_3_grid_search_analysis.ipynb`

- analyze different runs with varying hyperparameters on a single data set
- run for each protein group, peptides and precursor data set

### `02_4_best_models_over_all_data.ipynb`	

- show best models across data sets in grid search

### `03_1_best_models_comparison.ipynb`

## Misc

### `misc_clustering_proteins.ipynb`

- first PCA analysis of proteins from Annelaura

### `misc_data_exploration_proteins.ipynb` 

### `misc_embeddings.ipynb`

### `misc_illustrations.ipynb`
- illustrations for presentations
- e.g. shifted normal imputation

### `misc_pytorch_fastai_dataloaders.ipynb`

### `misc_pytorch_fastai_dataset.ipynb`
### `misc_id_mapper.ipynb`

### `misc_json_formats.ipynb`

### `run_ipynbs.py`

### `misc_protein_support.ipynb`

- map peptide sequences to protein sequences
- calculate some metrics

### `misc_sampling_in_pandas.ipynb`

### `misc_MaxQuantOutput.ipynb`
- misc

### 01 Analysis Fasta

#### `misc_FASTA_tryptic_digest.ipynb`

- analysis FASTA file used for protein search

#### `misc_FASTA_data_agg_by_gene.ipynb`

- analysis of gene to protein mapping of fasta file

### 02 Analysis dataset

#### `erda_data_available.ipynb`
- analyze `count_all_peptides.json`: How many peptides are identified overall in all
  processed files 

> erda notebook: `00_mq_count_peptides.ipynb`

#### `misc_data_exploration_peptides.ipynb` 
- finds files originationg from fractionation experiments
- plot mask indicating presence/abscence of peptide measurement in an experiment
- intensity log-transformation: 
