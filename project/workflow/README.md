# Snakemake workflows

> needs to be exectued from `project` directory (as root directory)

# Single experiment

```
snakemake -n # uses workflow/Snakefile
```

## Grid Search for Hyperparameters

```bash
# pwd: path/to/project
snakemake --snakefile workflow/Snakefile_GRID.smk -n -p
# parallel execution on the same machine (shared GPU, CPU) does not work
snakemake --snakefile workflow/Snakefile_GRID.smk --jobs 1
```

Configuration files are stored in `project/config` directory

provide one or more config files explicitly to overwrite defaults

```
snakemake --snakefile workflow/Snakefile_GRID.smk --configfile config/other.yaml -n -p
```

## Repeated training

### Repeated training of same models on same dataset

- see if model train stable on one dataset

```bash
snakemake --snakefile workflow/Snakefile_GRID.smk -n -p
```

### Repeated trainign of models across machine datasets

- see how model perform across similar datasets


```
snakemake --snakefile workflow\Snakefile_across_datasets.smk -p -c1 -n
```