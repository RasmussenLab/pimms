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
conda and pip to setup the environment. For a detailed description of a setup using conda,
see [instructions on setting up a virtual environment](docs/venv_setup.md).

Download the repository

```
git clone git@github.com:RasmussenLab/pimms.git
cd pimms
```

Using conda (or mamba), install the dependencies and the package in editable mode

```
# from main folder of repository (containing environment.yml)
conda env create -n pimms -f environment.yml # slower
mamba env create -n pimms -f environment.yml # faster, less then 5mins
```

> If on Mac M1: use  `environment_m1.yaml` where cudatoolkit is removed.

## Run Demo

Change to the [`project` folder](./project) and see it's [README](project/README.md)

> Currently there are only notebooks and scripts under `project`, 
> but shared functionality will be added under `vaep` folder-package: This can 
> then be imported using `import vaep`. See [`vaep/README.md`](vaep/README.md)

```
conda activate pimms # activate 
```

The demo will run an example on a small data set of 50 HeLa samples (protein groups):
  1. it describes the data and does create the splits based on the [example data](project/data/dev_datasets/HeLa_6070/README.md)
  2. it runs the three semi-supervised models next to some default heuristic methods
  3. it creates an comparison

The results are written to `./pimms/project/runs/example`, including `html` versions of the 
notebooks for inspection.

```
cd project
pwd # so be in ./pimms/project
snakemake -c1 -p -n # dryrun
snakemake -c1 -p
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


