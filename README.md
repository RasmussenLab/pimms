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
under the [`project`](project) folder.

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


## Setup for Development
The package is not yet available as a standalone software on pypi. Currently we use 
conda and pip to setup the environment. 

Using conda (or mamba), install the dependencies and the package in editable mode

```
# from main folder of repository (containing environment.yml)
conda create env -n pimms -f environment.yml # slower
mamba create env -n pimms -f environment.yml # faster
```

For a detailed description of a setup using conda, see [docs](docs/venv_setup.md)

> Currently there are only notebooks and scripts under `project`, 
> but shared functionality will be added under `vaep` folder-package: This can 
> then be imported using `import vaep`. See `vaep/README.md`

### Setup using pip

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

## Overview vaep package


