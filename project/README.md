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

Currently `pickle`d and `csv` files are supported. If you use csv files, make sure
to set an index name for the columns (default: `Sample ID`). It's done mostly automatically.

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
Tutorials | 
-  | 04_1_train_pimms_models.ipynb | main tutorial showing scikit-learn (Transformer) interface partly with or without validation data
Single experiment |
run  | 01_0_split_data.ipynb               | Create train, validation and test data splits
run  | 01_0_transform_data_to_wide_format.ipynb | Transform train split to wide format for R models
run  | 01_1_train_<model>.ipynb            | Train a single model e.g. (VAE, DAE, CF)
run  | 01_1_train_NAGuideR_methods.ipynb   | Train supported R models
run  | 01_1_transfer_NAGuideR_pred.ipynb   | Transfer R model predictions to correct format in Python
run  | 01_2_performance_plots.ipynb        | Performance of single model run
Grid search and best model analysis |
grid | 02_1_{aggregate|join}_metrics.py.ipynb    | Aggregate or join metrics
grid | 02_2_{aggregate|join}_configs.py.ipynb    | Aggregate or join model configurations
grid | 02_3_grid_search_analysis.ipynb    | Analyze different runs with varying hyperparameters on a dataset
grid | 02_4_best_models_over_all_data     | Show best models and best models across data types
best | 03_1_best_models_comparison.ipynb  | best model trained repeatedly or across datasets
Differential analysis workflow |
ald | 10_0_ald_data.ipynb               | preprocess data -> could be move to data folder
ald | 10_1_ald_diff_analysis.ipynb      | differential analysis (DA), dump scores 
ald | 10_2_ald_compare_methods.ipynb    | DA comparison between methods
ald | 10_3_ald_ml_new_feat.ipynb        | ML model comparison
ald | 10_4_ald_compare_single_pg.ipynb  | Compare imputation for feat between methods (dist plots)
ald | 10_5_comp_diff_analysis_repetitions.ipynb | [Not in workflow] Compare 10x repeated differential analysis workflow
ald | 10_6_interpret_repeated_ald_da.py | [Not in workflow] Interpret 10x repeated differential analysis
ald | 10_7_ald_reduced_dataset_plots.ipynb | [Not in workflow] Plots releated reduced dataset (80% dataset)
Data inspection and manipulations for experiments |
data | 00_5_training_data_exploration.py | Inspect dataset
data | 00_6_0_permute_data.ipynb | Permute data per column to check overfitting of models (mean unchanged per column)
data | 00_8_add_random_missing_values.py | Script to add random missing values to ALD data
Publication specific notebooks |
pub | 03_2_best_models_comparison_fig2.ipynb | Best models comparison in Fig. 2
pub | 03_3_combine_experiment_result_tables.ipynb | Combine HeLa experiment results for reporting
pub | 03_4_join_tables.py | Combine ALD experiment results for reporting
pub | 03_6_setup_comparison_rev3.py | Analyze setup of KNN comparison for rev 3
Miscancellous notebooks on different topics (partly exploration) |
misc | misc_embeddings.ipynb                | FastAI Embeddings
misc | misc_illustrations.ipynb             | Illustrations of certain concepts (e.g. draw from shifted random distribution)
misc | misc_json_formats.ipynb              | Investigate storring training data as json with correct encoding
misc | misc_pytorch_fastai_dataset.ipynb    | Dataset functionality
misc | misc_pytorch_fastai_dataloaders.ipynb| Dataloading functionality
misc | misc_sampling_in_pandas.ipynb        | How to sample in pandas

## KNN adhoc analysis using jupytext and papermill

Compare performance splitting samples into train, validation and test set.
Use scikit-learn `KNN_IMPUTER` as it's easiest to tweak and understand.

```bash
# classic:
jupytext --to ipynb -k - -o - 01_1_train_KNN.py | papermill - runs/rev3/01_1_train_KNN.ipynb
# train only on samples without simulated missing values, add simulated missing values to test and validation samples
jupytext --to ipynb -k - -o - 01_1_train_KNN_unique_samples.py | papermill - runs/rev3/01_1_train_KNN_unique_samples.ipynb
# new comparison (check if the old nb could be used for this purpose)
jupytext --to ipynb -k - -o - 01_3_revision3.py | papermill - runs/rev3/01_3_revision3.ipynb
```