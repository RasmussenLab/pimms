# Config files

## Version 1 imputation workflow

For [`worflow/Snakefile`](https://github.com/RasmussenLab/pimms/blob/HEAD/project/workflow/Snakefile)

```bash
config.yaml # main config
split.yaml # split data config referenced in config.yaml
train_CF.yaml # CF train config referenced in config.yaml
train_DAE.yaml # DAE train config referenced in config.yaml
train_KNN.yaml # KNN train config referenced in config.yaml
train_Median.yaml # Median train config referenced in config.yaml
train_VAE.yaml # VAE train config referenced in config.yaml
```

## Version 2 impuation workflow

For [`workflow/Snakefile_v2.yaml`](https://github.com/RasmussenLab/pimms/blob/HEAD/project/workflow/Snakefile_v2.smk) only one config file is needed:

```bash
config_v2.yaml
```