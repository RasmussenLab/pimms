# PIMMS

PIMMS stands for Proteomics Imputation Modeling Mass Spectrometry 
and is a hommage to our dear British friends 
who are missing as part of the EU for far too long already.
(Pimms is also a british summer drink)


> `PIMMS`was called `vaep` during development.  
> Before entire refactoring has to been completed the imported package will be
`vaep`.

## Setup
Clone repository and install package in editable mode:

```
pip install -e /path/to/cloned/folder 
```

For a detailed description of a setup using conda, see [docs](docs/venv.md)

> Currently there are only notebooks and scripts under `project`, 
> but shared functionality will be added under `vaep` folder-package: This can 
> then be imported using `import vaep`. See `vaep/README.md`

## Overview vaep package
- imputation of data is done based on the standard variation or KNN imputation. Adapted scripts from Annelaura are under `vaep/imputation.py`
- transformation of intensity data is under `vaep/transfrom.py`


