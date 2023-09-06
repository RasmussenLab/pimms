# PIMMS

PIMMS stands for Proteomics Imputation Modeling Mass Spectrometry 
and is a hommage to our dear British friends 
who are missing as part of the EU for far too long already
(Pimms is also a British summer drink).

The pre-print is available [on biorxiv](https://doi.org/10.1101/2023.01.12.523792).


> `PIMMS` was called `vaep` during development.  
> Before entire refactoring has to been completed the imported package will be
`vaep`.

We provide functionality as a python package, an excutable workflow and notebooks.

The models can be used with the scikit-learn interface in the spirit of other scikit-learn imputers. You can try this in colab. [![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/pimms/blob/dev/project/04_1_train_pimms_models.ipynb)



## Notebooks as scripts using papermill

If you want to run a model on your prepared data, you can run notebooks prefixed with 
`01_`, i.e. `project/01_*.ipynb`. Using jupytext also python percentage script versions
are saved.

```
cd project # project folder as pwd
papermill 01_0_split_data.ipynb --help-notebook
papermill 01_1_train_vae.ipynb --help-notebook
```

> Misstyped argument names won't throw an error when using papermill

### Python package

For interactive use of the models provided in PIMMS, you can use our
[python package `pimms-learn`](https://pypi.org/project/pimms-learn/).
The interface is similar to scikit-learn.


```
pip install pimms-learn
```


Then you can use the models on a pandas DataFrame with missing values. Try this in the tutorial on Colab:
[![open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RasmussenLab/pimms/blob/dev/project/04_1_train_pimms_models.ipynb)


## Setup for PIMMS comparison workflow

The package is available as a standalone software on pypi. However, running the entire snakemake workflow in enabled using 
conda (or mamba) and pip to setup the environment. For a detailed description of setting up
conda (or mamba), see [instructions on setting up a virtual environment](docs/venv_setup.md).

Download the repository

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

If on Mac M1, M2 or having otherwise issue using your accelerator (e.g. GPUs): Install the pytorch dependencies first, then the rest of the environment.

### Install development dependencies

Check how to install pytorch for your system [here](https://pytorch.org/get-started/previous-versions/#v1131).

- select the version compatible with your cuda version if you have an nvidia gpu

```bash
conda create -n vaep_manuel python=3.8 pip
conda activate vaep_manuel
conda update pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia # might be different
pip install . # pimms-learn
pip install papermill jupyterlab # use run notebook interactive or as a script
cd project
papermill 04_1_train_pimms_models.ipynb 04_1_train_pimms_models_test.ipynb # second notebook is output
python 04_1_train_pimms_models.ipynb # just execute the code
# jupyter lab # open 04_1_train_pimms_models.ipynb
```

### Entire development installation


```bash
conda create -n pimms_dev -c pytorch -c nvidia -c fastai -c bioconda -c plotly -c conda-forge --file requirements.txt --file requirements_R.txt --file requirements_dev.txt
pip install -e . # other pip dependencies missing
snakemake --configfile config/single_dev_dataset/example/config.yaml -F -n
```

or if you want to update an existing environment


```
conda update  -c defaults -c conda-forge -c fastai -c bioconda -c plotly --file requirements.txt --file requirements_R.txt --file requirements_dev.txt
```

or using the environment.yml file (can fail on certain systems)

```
conda env create -f environment.yml
```


### Troubleshooting

Trouble shoot your R installation by opening jupyter lab

```
# in projects folder
jupyter lab # open 01_1_train_NAGuideR.ipynb

## Run Demo

Change to the [`project` folder](./project) and see it's [README](project/README.md)

> Currently there are only notebooks and scripts under `project`, 
> but shared functionality will be added under `vaep` folder-package: This can 
> then be imported using `import vaep`. See [`vaep/README.md`](vaep/README.md)

You can subselect models by editing the config file:  [`config.yaml`](project/config/single_dev_dataset/proteinGroups_N50/config.yaml) file.

```
conda activate pimms # activate virtual environment
cd project # go to project folder
pwd # so be in ./pimms/project
snakemake -c1 -p -n # dryrun demo workflow
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
| ------------- | ----------------- | ------       | --- |------------------ | 
| CF            | pimms             | pip          | | Collaborative Filtering |
| DAE           | pimms             | pip          | | Denoising Autoencoder   |
| VAE           | pimms             | pip          | | Variational Autoencoder |     
|  |   | | | 
| ZERO          | -                 | -            | | replace NA with 0 |
| MINIMUM       | -                 | -            | | replace NA with global minimum    |
| COLMEDIAN     | e1071             | CRAN         | | replace NA with column median  |
| ROWMEDIAN     | e1071             | CRAN         | | replace NA with row median     |
| KNN_IMPUTE    | impute            | BIOCONDUCTOR | | k nearest neighbor imputation   |
| SEQKNN        | SeqKnn            | tar file     | | Sequential k- nearest neighbor imputation <br> start with feature with least missing values and re-use imputed values for not yet imputed features
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
| ~~grr~~       | DreamAI           | -            | Fails to install | Rigde regression 
| ~~GMS~~       | GMSimpute         | tar file     | Fails on Windows | Lasso model



## Workflows

The workflows folder in the repository contains snakemake workflows used for rawfile data processing, 
both for running MaxQuant over a large set of HeLa raw files 
and ThermoRawFileParser on a list of raw files to extract their meta data. For details see:

>  Webel, Henry, Yasset Perez-Riverol, Annelaura Bach Nielson, and Simon Rasmussen. 2023. “Mass Spectrometry-Based Proteomics Data from Thousands of HeLa Control Samples.” Research Square. https://doi.org/10.21203/rs.3.rs-3083547/v1.

### MaxQuant

Process single raw files using MaxQuant. See [README](workflows/maxquant/README.md) for details.

### Metadata

Read metadata from single raw files using MaxQuant. See [README](workflows/metadata/README.md) for details.

## Build status
[![Documentation Status](https://readthedocs.org/projects/pimms/badge/?version=latest)](https://pimms.readthedocs.io/en/latest/?badge=latest)