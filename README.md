# PIMMS

PIMMS stands for Proteomics Imputation Modeling Mass Spectrometry 
and is a hommage to our dear British friends 
who are missing as part of the EU for far too long already.
(Pimms is also a british summer drink)

The pre-print is available [on biorxiv](https://www.biorxiv.org/content/10.1101/2023.01.12.523792v1).


> `PIMMS`was called `vaep` during development.  
> Before entire refactoring has to been completed the imported package will be
`vaep`.

We provide functionality as a python package and excutable workflows and notebooks 
under the [`project`](project) folder, inclduing an example.

The [`workflows`](workflows) folder contains snakemake workflows used for rawfile data processing, 
both for [running MaxQuant](workflows\maxquant) over a large set of HeLa raw files 
and ThermoRawFileParser on a list of raw files to [extract their meta data](workflows\metadata).

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

### Outlook

We also plan to provide functionality and examples to interactive use of the 
models developed in PIMMS.

## Setup
The package is not yet available as a standalone software on pypi. Currently we use 
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

If on Mac M1: use  `environment_m1.yaml` where cudatoolkit is removed.

```
conda env create -n pimms -f environment_m1.yml # slower
mamba env create -n pimms -f environment_m1.yml # faster, less then 5mins
```

If on Windows: use `environment_win.yaml` where ~~two R-Bioconductor~~ R-packages (see note bolow) are removed as 
no binaries are available for Windows. You will need to install these manually afterwards if you want to use methods implemented in R.

> Note: Turns out that installing dependencies partly by conda and partly manuaelly
using `BiocManager` is not working.

```
conda env create -n pimms -f environment_win.yml # slower
mamba env create -n pimms -f environment_win.yml # faster, less then 5mins
# Then if R packages are needed, they are installed on the fly for Windows.
# Could be used as well for MacOS or Linux.
```

Trouble shoot your R installation by opening jupyter lab

```
# in projects folder
jupyter lab # open 01_1_train_NAGuideR.ipynb
```

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



<!-- ### Setup using pip

> Dependecies are currently provided through `environment.yml`, see above

From GitHub
```
pip install git+https://github.com/RasmussenLab/pimms.git
```

Using the clone repository
```
pip install /path/to/cloned/folder 
```

And using the cloned repository for an editable installation
```
pip install -e /path/to/cloned/folder 
```

## Overview vaep package -->


