# Paper project

## Data requirements

(tbc)

ThermoRawFileParser output is used as metadata so far (a workflow is provided).
Strictly required is an ordering meta data information,e.g. a measurment data, and whatever else
   (e.g. some clinical metadata over samples)
- `meta_date_col`: currently stricktly required
- `meta_cat_col`: optional column used for visualization of PCAs

## Workflows

> snakefile stored in [workflow](workflow/README.md)

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

```cmd
# cwd: project folder (this folder)
snakemake --configfile config/single_dev_dataset/aggpeptides/config.yaml -p -n 
```

### Single notebooks using papermill

execute single notebooks
```cmd
set DATASET=df_intensities_proteinGroups_long/Q_Exactive_HF_X_Orbitrap_6070 
papermill  01_0_split_data.ipynb --help-notebook # check parameters
papermill  01_0_split_data.ipynb runs/experiment_03/%DATASET%/experiment_03_data.ipynb -p MIN_SAMPLE 0.5 -p fn_rawfile_metadata data/dev_datasets/%DATASET%.csv -p index_col "Sample ID" -p columns_name peptide
```

## Notebooks

- erda: Is the longterm storage of the university -> MQ output was processed on a server attached to erda
- hela: dumps from erda processing (raw file names, aggregated `summaries.txt` from MQ, protein groups, peptides and precursor dumps)
- run: a single experiment with models attached, see `workflow/Snakefile`
- grid: only grid search associated, see `workflow/Snakefile_grid.smk`
- best: best models repeatedly trained or across datasets, see `workflow/Snakefile_best_repeated.smk` and `workflow/Snakefile_best_across_datasets.smk`
- ald: ALD study associated, see `workflow/Sankefile_ald_comparison.smk`

tag | notebook  | Description
--- | ---  |  --- 
Development data related 
erda | erda_01_mq_select_runs.ipynb         | Aggregate current summary files from MQ runs into table
erda | erda_02_mq_count_features.ipynb      | Aggregate information from all eligable MQ runs <br> Saves processed files used for data selection (Counters used in `erda_03_training_data.ipynb`)
erda | erda_03_training_data.ipynb          | Build training data dump (run for each data level) in wide format
erda | erda_04_transpose_data.ipynb         | Transpose dataset (row: a sample), separate as erda has memory limits, dump counts and present-absent patterns
erda | erda_12_explore_raw_MQ_data.ipynb    | Load a single MQ txt output folder and browse data <br> dumps large pickle files for training
erda | erda_data_available.ipynb            | Plots on theoretically available data based on Counter dictionaries
hela | 00_0_hela_metadata_rawfiles.ipynb         |  Analyze rawfile metadata and prepare for data selection
hela | 00_1_hela_MQ_summaries.ipynb              | Analyzse summaries.txt data from all samples
hela | 00_2_hela_all_raw_files.ipynb             | Find duplicate raw files, analyze sizes
hela | 00_3_hela_selected_files_overview.ipynb   | Data description based on file size and metaddata of selected files
hela | 00_4_hela_development_dataset_splitting   | Splitting data into development datasets of HeLa cell line data (based on wide format input from `erda_03` and `erda_04`)
Single development dataset |
hela | 00_5_hela_development_dataset_support.ipynb    | Support of training data samples/feat on selected development data set
hela | 00_6_hela_training_data_exploration.ipynb  | Explore a data set for diagnositics <br>  Visualize key metrics
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
misc | misc_clustering_proteins.ipynb       | \[documentation\] PCA protein analysis from Annelaura w/ initial data <br> (Executed, only for documentation)
misc | misc_data_exploration_peptides.ipynb | Describe current peptides training data
misc | misc_data_exploration_proteins.ipynb | \[NEEDS UPDATE\] Describe small initial protein training data 
misc | misc_embeddings.ipynb                | FastAI Embeddings
misc | misc_FASTA_data_agg_by_gene.ipynb    | Investigate possibility to join proteins by gene
misc | misc_FASTA_tryptic_digest.ipynb      | Analyze fasta file used for peptide identification
misc | misc_id_mapper.ipynb                 | train models per gene, see overlaps in proteins, see coverage   | of proteins with observed peptides, align overlapping peptide sequences
misc | misc_illustrations.ipynb             | Illustrations of certain concepts (e.g. draw from shifted random distribution)
misc | misc_json_formats.ipynb              | Investigate storring training data as json with correct encoding
misc | misc_MaxQuantOutput.ipynb            | \[documentation\] Analyze MQ output, show MaxQuantOutput class behaviour
misc | misc_protein_support.ipynb           | peptide sequences mapped to protein sequences
misc | misc_pytorch_fastai_dataset.ipynb    | Dataset functionality
misc | misc_pytorch_fastai_dataloaders.ipynb| Dataloading functionality
misc | misc_sampling_in_pandas.ipynb        | How to sample in pandas

# Notebook descriptions (To be completed)

## erda notebooks

- [ ] determine order and rename accordingly with prefix

The data is for now processed only using MaxQuant. If the files are processed
by another Software, these notebooks need to be adapted for if they contain `mq` or `MQ`.

### erda_01_mq_select_runs

- read in all summaries and select eligable runs based on number of identified peptides

### erda_02_mq_count_features

- Feature Extraction and Feature counting
- dumps extracted features per group into `FOLDER_PROCESSED`
  (separated for type and by year)

### erda_03_training_data

- needs to be executed for each data type
- loads a python config file (setting `FeatureCounter` classes and custom functions)
  along string configuration variables

## HeLa notebooks - Training data


### `00_0_1_rawfile_renaming.ipynb`

> internal, documentation only (see pride upload for result)

- create a new id for each raw file based on the creation date and instrument
- uses metadata
- build lftp commands for pride upload

### `00_0_hela_metadata_rawfiles.ipynb`

- group by MS instrument parameters
- create `data/files_per_instrument_nested.yaml` for selection of data by massspectrometer

### `00_1_hela_MQ_summaries.ipynb`

- analysze all `summaries.txt`

### `00_2_hela_all_raw_files.ipynb`

### `00_3_hela_selected_files_overview.ipynb`

- created joined metadata file 
- overview of metadata of selected files for data descriptor paper

### `00_4_hela_development_dataset_splitting.ipynb`

- Create development dataset(s) of common machines, one for each machine
- UMAP **Figure 1b**, statistics of **Figure 1c**
- create datasets for training PIMMS models

### Training data inspection

### `00_5_hela_development_dataset_support.ipynb`

- feature counts for a single development dataset (e.g. for a single machine)

### `00_6_hela_training_data_exploration.ipynb`

> needs clean-up

- explore a data set for diagnositics

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
