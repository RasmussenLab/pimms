# Snakemake workflows

> needs to be started from `project` directory (as root directory)


```bash
# pwd: path/to/project
snakemake --snakefile Snakefile_GRID.smk -n -p
# parallel execution on the same machine (shared GPU, CPU) does not work
snakemake --snakefile Snakefile_GRID.smk --jobs 1
```
