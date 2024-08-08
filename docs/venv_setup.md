# Setting up an Virtual Environment

## 1. Installing conda

### Miniforge

Mamba is conda, only implemented not in Python: Write `mamba` where it says `conda`, and 
it should work exactly the same.

Fresh install instruction [here](https://github.com/conda-forge/miniforge)

### Miniconda

Fresh install instructions [here](https://docs.conda.io/en/latest/miniconda.html)

## 2. Create Conda env

Installing the development version, run 
```
conda env create -f environment.yml
mamba env create -f environment.yml
```
