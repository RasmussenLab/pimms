![log](docs/logos/logo.png)
[![Read the Docs](https://img.shields.io/readthedocs/pimms)](https://readthedocs.org/projects/pimms/) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/RasmussenLab/pimms/ci.yaml)](https://github.com/RasmussenLab/pimms/actions) [![Documentation Status](https://readthedocs.org/projects/pimms/badge/?version=latest)](https://pimms.readthedocs.io/en/latest/?badge=latest)

PIMMS stands for Proteomics Imputation Modeling Mass Spectrometry 
and is a hommage to our dear British friends 
who are missing as part of the EU for far too long already
(Pimms is a British summer drink).

We published the [work](https://www.nature.com/articles/s41467-024-48711-5) in Nature Communications as open access: 

> Webel, H., Niu, L., Nielsen, A.B. et al.  
> Imputation of label-free quantitative mass spectrometry-based proteomics data using self-supervised deep learning.  
> Nat Commun 15, 5405 (2024).  
> https://doi.org/10.1038/s41467-024-48711-5

We provide functionality as a python package, an excutable workflow or simply in notebooks.

For any questions, please [open an issue](https://github.com/RasmussenLab/pimms/issues) or contact me directly.

## Getting started

The models can be used with the scikit-learn interface in the spirit of other scikit-learn imputers. You can try this using our tutorial in colab:

[![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/pimms/blob/HEAD/project/04_1_train_pimms_models.ipynb)

It uses the scikit-learn interface. The PIMMS models in the scikit-learn interface
can be executed on the entire data or by specifying a valdiation split for checking training process.
In our experiments overfitting wasn't a big issue, but it's easy to check.

## Install Python package

For interactive use of the models provided in PIMMS, you can use our
[python package `pimms-learn`](https://pypi.org/project/pimms-learn/).
The interface is similar to scikit-learn.

```
pip install pimms-learn
```

Then you can use the models on a pandas DataFrame with missing values. You can try this in the tutorial on Colab by uploading your data:
[![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/pimms/blob/HEAD/project/04_1_train_pimms_models.ipynb)

> `PIMMS` was called `vaep` during development.  
> Before entire refactoring has been completed the imported package will be `vaep`.

## Notebooks as scripts using papermill

If you want to run a model on your prepared data, you can run notebooks prefixed with 
`01_`, i.e. [`project/01_*.ipynb`](https://github.com/RasmussenLab/pimms/tree/HEAD/project) after cloning the repository. Using jupytext also python percentage script versions
are saved.

```bash
# navigat to your desired folder
git clone https://github.com/RasmussenLab/pimms.git # get all notebooks
cd project # project folder as pwd
# pip install pimms-learn papermill # if not already installed
papermill 01_0_split_data.ipynb --help-notebook
papermill 01_1_train_vae.ipynb --help-notebook
```
> :warning: Mistyped argument names won't throw an error when using papermill, but a warning is printed on the console thanks to my contributions:)

## PIMMS comparison workflow and differential analysis workflow

The PIMMS comparison workflow is a snakemake workflow that runs the all selected PIMMS models and R-models on 
a user-provided dataset and compares the results. An example for a publickly available Alzheimer dataset on the 
protein groups level is re-built regularly and available at: [rasmussenlab.org/pimms](https://www.rasmussenlab.org/pimms/)

It is built on top of
  - the [Snakefile_v2.smk](https://github.com/RasmussenLab/pimms/blob/HEAD/project/workflow/Snakefile_v2.smk) (v2 of imputation workflow), specified in on configuration
  - the [Snakefile_ald_comparision](https://github.com/RasmussenLab/pimms/blob/HEAD/project/workflow/Snakefile_ald_comparison.smk) workflow for differential analysis

The associated notebooks are index with `01_*` for the comparsion workflow and `10_*` for the differential analysis workflow. The `project` folder can be copied separately to any location if the package is installed. It's standalone folder. It's main folders are:

```bash
# project folder:
project
│   README.md # see description of notebooks and hints on execution in project folder
|---config # configuration files for experiments ("workflows")
|---data # data for experiments
|---runs # results of experiments
|---src # source code or binaries for some R packges
|---tutorials # some tutorials for libraries used in the project
|---workflow # snakemake workflows
```

To re-execute the entire workflow locally, have a look at the [configuration files](https://github.com/RasmussenLab/pimms/tree/HEAD/project/config/alzheimer_study) for the published Alzheimer workflow:

- [`config/alzheimer_study/config.yaml`](https://github.com/RasmussenLab/pimms/blob/HEAD/project/config/alzheimer_study/comparison.yaml)
- [`config/alzheimer_study/comparsion.yaml`](https://github.com/RasmussenLab/pimms/blob/HEAD/project/config/alzheimer_study/config.yaml)

To execute that workflow, follow the Setup instructions below and run the following command in the project folder:

```bash
# being in the project folder
snakemake -s workflow/Snakefile_v2.smk --configfile config/alzheimer_study/config.yaml -p -c1 -n # one core/process, dry-run
snakemake -s workflow/Snakefile_v2.smk --configfile config/alzheimer_study/config.yaml -p -c2 # two cores/process, execute
# after imputation workflow, execute the comparison workflow
snakemake -s workflow/Snakefile_ald_comparison.smk --configfile config/alzheimer_study/comparison.yaml -p -c1
# If you want to build the website locally: https://www.rasmussenlab.org/pimms/
pip install .[docs]
pimms-setup-imputation-comparison -f project/runs/alzheimer_study/
pimms-add-diff-comp -f project/runs/alzheimer_study/ -sf_cp project/runs/alzheimer_study/diff_analysis/AD
cd project/runs/alzheimer_study/
sphinx-build -n --keep-going -b html ./ ./_build/
# open ./_build/index.html
```

## Setup workflow and development environment

### Setup comparison workflow

The core funtionality is available as a standalone software on PyPI under the name `pimms-learn`. However, running the entire snakemake workflow in enabled using 
conda (or mamba) and pip to setup an analysis environment. For a detailed description of setting up
conda (or mamba), see [instructions on setting up a virtual environment](https://github.com/RasmussenLab/pimms/blob/HEAD/docs/venv_setup.md).

Download the repository:

```
git clone https://github.com/RasmussenLab/pimms.git
cd pimms
```

Using conda (or mamba), install the dependencies and the package in editable mode

```
# from main folder of repository (containing environment.yml)
conda env create -n pimms -f environment.yml # slower
mamba env create -n pimms -f environment.yml # faster, less then 5mins
```

If on Mac M1, M2 or having otherwise issue using your accelerator (e.g. GPUs): Install the pytorch dependencies first, then the rest of the environment:

### Install pytorch first

> :warning: We currently see issues with some installations on M1 chips. A dependency
> for one workflow is polars, which causes the issue. This should be [fixed now](https://github.com/RasmussenLab/njab/pull/13) 
> for general use by delayed import 
> of `mrmr-selection` in `njab`. If you encounter issues, please open an issue.

Check how to install pytorch for your system [here](https://pytorch.org/get-started).

- select the version compatible with your cuda version if you have an nvidia gpu or a Mac M-chip.

```bash
conda create -n pimms python=3.9 pip
conda activate pimms
# Follow instructions on https://pytorch.org/get-started: 
# CUDA is not available on MacOS, please use default package
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda install pytorch::pytorch torchvision torchaudio fastai -c pytorch -c fastai -y
pip install pimms-learn
pip install jupyterlab papermill # use run notebook interactively or as a script

cd project
# choose one of the following to test the code
jupyter lab # open 04_1_train_pimms_models.ipynb
papermill 04_1_train_pimms_models.ipynb 04_1_train_pimms_models_test.ipynb # second notebook is output
python 04_1_train_pimms_models.py # just execute the code
```

### Let Snakemake handle installation

If you only want to execute the workflow, you can use snakemake to build the environments for you:

> Snakefile workflow for imputation v1 only support that atm.

```bash
snakemake -p -c1 --configfile config/single_dev_dataset/example/config.yaml --use-conda -n # dry-run
snakemake -p -c1 --configfile config/single_dev_dataset/example/config.yaml --use-conda # execute with one core
```

### Troubleshooting

Trouble shoot your R installation by opening jupyter lab

```
# in projects folder
jupyter lab # open 01_1_train_NAGuideR.ipynb
```

## Run example

Change to the [`project` folder](./project) and see it's [README](project/README.md)
You can subselect models by editing the config file:  [`config.yaml`](https://github.com/RasmussenLab/pimms/tree/HEAD/project/config/single_dev_dataset/proteinGroups_N50) file.

```
conda activate pimms # activate virtual environment
cd project # go to project folder
pwd # so be in ./pimms/project
snakemake -c1 -p -n # dryrun demo workflow, potentiall add --use-conda
snakemake -c1 -p
```

The demo will run an example on a small data set of 50 HeLa samples (protein groups):
  1. it describes the data and does create the splits based on the [example data](project/data/dev_datasets/HeLa_6070/README.md)
     - see `01_0_split_data.ipynb`
  2. it runs the three semi-supervised models next to some default heuristic methods
     - see `01_1_train_collab.ipynb`, `01_1_train_dae.ipynb`, `01_1_train_vae.ipynb`
  3. it creates an comparison
     - see `01_2_performance_plots.ipynb`

The results are written to `./pimms/project/runs/example`, including `html` versions of the 
notebooks for inspection, having the following structure:

```
│   01_0_split_data.html
│   01_0_split_data.ipynb
│   01_1_train_collab.html
│   01_1_train_collab.ipynb
│   01_1_train_dae.html
│   01_1_train_dae.ipynb
│   01_1_train_vae.html
│   01_1_train_vae.ipynb
│   01_2_performance_plots.html
│   01_2_performance_plots.ipynb
│   data_config.yaml
│   tree_folder.txt
|---data
|---figures
|---metrics
|---models
|---preds
```

The predictions of the three semi-supervised models can be found under `./pimms/project/runs/example/preds`.
To combine them with the observed data you can run

```python
# ipython or python session
# be in ./pimms/project
folder_data = 'runs/example/data'
data = vaep.io.datasplits.DataSplits.from_folder(
    folder_data, file_format='pkl')
observed = pd.concat([data.train_X, data.val_y, data.test_y])
# load predictions for missing values of a certain model
model = 'vae'
fpath_pred = f'runs/example/preds/pred_real_na_{model}.csv '
pred = pd.read_csv(fpath_pred, index_col=[0, 1]).squeeze()
df_imputed = pd.concat([observed, pred]).unstack()
# assert no missing values for retained features
assert df_imputed.isna().sum().sum() == 0
df_imputed
```

## Available imputation methods

Packages either are based on this repository, or were referenced by NAGuideR (Table S1).
From the brief description in the table the exact procedure is not always clear.

| Method        | Package           | source       | status | name              |
| ------------- | ----------------- | ------       | ------ |------------------ | 
| CF            | pimms             | pip          | | Collaborative Filtering |
| DAE           | pimms             | pip          | | Denoising Autoencoder   |
| VAE           | pimms             | pip          | | Variational Autoencoder |     
|  |   | | | 
| ZERO          | -                 | -            | | replace NA with 0 |
| MINIMUM       | -                 | -            | | replace NA with global minimum    |
| COLMEDIAN     | e1071             | CRAN         | | replace NA with column median  |
| ROWMEDIAN     | e1071             | CRAN         | | replace NA with row median     |
| KNN_IMPUTE    | impute            | BIOCONDUCTOR | | k nearest neighbor imputation   |
| SEQKNN        | SeqKnn            | tar file     | | Sequential k- nearest neighbor imputation <br> starts with feature with least missing values and re-use imputed values for not yet imputed features
| BPCA          | pcaMethods        | BIOCONDUCTOR | | Bayesian PCA missing value imputation
| SVDMETHOD     | pcaMethods        | BIOCONDUCTOR | | replace NA initially with zero, use k most significant eigenvalues using Singular Value Decomposition for imputation until convergence
| LLS           | pcaMethods        | BIOCONDUCTOR | | Local least squares imputation of a feature based on k most correlated features
| MLE           | norm              | CRAN         | | Maximum likelihood estimation
| QRILC         | imputeLCMD        | CRAN         | | quantile regression imputation of left-censored data, i.e. by random draws from a truncated distribution which parameters were estimated by quantile regression
| MINDET        | imputeLCMD        | CRAN         | | replace NA with q-quantile minimum in a sample
| MINPROB       | imputeLCMD        | CRAN         | | replace NA by random draws from q-quantile minimum centered distribution
| IRM           | VIM               | CRAN         | | iterativ robust model-based imputation (one feature at at time)
| IMPSEQ        | rrcovNA           | CRAN         | | Sequential imputation of missing values by minimizing the determinant of the covariance matrix with imputed values
| IMPSEQROB     | rrcovNA           | CRAN         | | Sequential imputation of missing values using robust estimators
| MICE-NORM     | mice              | CRAN         | | Multivariate Imputation by Chained Equations (MICE) using Bayesian linear regression
| MICE-CART     | mice              | CRAN         | | Multivariate Imputation by Chained Equations (MICE) using regression trees
| TRKNN         | -                 | script       | | truncation k-nearest neighbor imputation 
| RF            | missForest        | CRAN         | | Random Forest imputation (one feature at a time)
| PI            | -                 | -            | | Downshifted normal distribution (per sample)
| GSIMP         | -                 | script       | | QRILC initialization and iterative Gibbs sampling with generalized linear models (glmnet)
| MSIMPUTE      | msImpute          | BIOCONDUCTOR | | Missing at random algorithm using low rank approximation
| MSIMPUTE_MNAR | msImpute          | BIOCONDUCTOR | | Missing not at random algorithm using low rank approximation
| ~~grr~~       | DreamAI           | -            | Fails to install | Rigde regression 
| ~~GMS~~       | GMSimpute         | tar file     | Fails on Windows | Lasso model
