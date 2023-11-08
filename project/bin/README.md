# Computerome2 (CR2) scripts

Cluster exectuion script for CR2 using a torque-pbs queue.


## Distributed

```bash
qsub run_snakemake_cluster.sh -N snakemake_exp -v configfile=config/single_dev_dataset/example/config.yaml,prefix=exp
```

## Single node

```bash
qsub run_snakemake.sh -N grid_exp
```


