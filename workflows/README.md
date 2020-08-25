# Workflows

## Snakemake
Snakemake is a framework for execution of workflows on UNIX based systems.
It is written in the line of thought of 
[`GNU Makefiles`](https://www.opensourceforu.com/2012/06/gnu-make-in-detail-for-beginners/),
but as an extension to Python rather than `C/C++`.

### Setup
```
conda install -n snakemake snakemake pygraphviz
```

## Downloading from erda

### Manually
In your `~/.ssh/config` define a target, here the `SharedFolderName` is called by `hela`:
```
Host hela
Hostname io.erda.dk
VerifyHostKeyDNS yes
User SharedFolderName
```

Then you can connect to `hela` using the `sftp`-command, and copy files to your
local `data`-folder:
```
sftp -B 258048 hela # pw is SharedFolderName
get file1.raw data/
get file2.raw data/
```

### In Shell Script

### In Python Script
Checkout snakemakes [SFTP](https://snakemake.readthedocs.io/en/stable/snakefiles/remote_files.html#file-transfer-over-ssh-sftp)
functionality which uses [`pysftp`](https://pysftp.readthedocs.io/en/release_0.2.8/pysftp.html#pysftp.Connection).
