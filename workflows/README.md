# Workflows

[Computerome](https://www.computerome.dk/display/CW/Batch+System)
 is based TORQUE `qsub` implementation.

#### Snakemake
- [qsub](https://snakemake.readthedocs.io/en/stable/executing/cluster-cloud.html#cluster-execution) execution
#### Nextflow
- [`PBS/TORQUE`](https://www.nextflow.io/docs/latest/executor.html#pbs-torque) execution

## MaxQuant workflow
> Currently single python script


## Downloading from erda

### Manually
```
# .ssh/config
Host hela
Hostname io.erda.dk
VerifyHostKeyDNS yes
User SharedFolderName
```


```
sftp -B 258048 hela # pw is SharedFolderName
get 20191028_QX4_StEb_MA_HeLa_500ng_191029155608.raw data/
get 20191028_QX3_LiSc_MA_Hela_500ng_LC15_1.raw data/
```

### In Shell Script

### In Python Script
Checkout snakemakes [SFTP](https://snakemake.readthedocs.io/en/stable/snakefiles/remote_files.html#file-transfer-over-ssh-sftp)
functionality which uses [`pysftp`](https://pysftp.readthedocs.io/en/release_0.2.8/pysftp.html#pysftp.Connection).
