# Setting up an Virtual Environment

## Conda env

Installing the development version, run 
```
conda env create -f environment.yml
```
and register the Ipython Kernel (alternatively install the whole Jupyter Suite
in the virtual env)
```
python -m ipykernel install --user --name other-env --display-name "Python (other-env)"
``` 

## Virtual env
