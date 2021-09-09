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

## Interacting with erda

### Setup
In your `~/.ssh/config` define a target, here the `SharedFolderName` is called by `hela`:

```
Host hela
Hostname io.erda.dk
VerifyHostKeyDNS yes
User SharedFolderName
```

### Connect interactively

Then you can connect to `hela` using the `sftp`-command, and copy files to your
local `data`-folder:

```
sftp -B 258048 hela # pw is SharedFolderName
get file1.raw data/
get file2.raw data/
```

### In Shell Script

### In Python Script
Checkout snakemake's [SFTP](https://snakemake.readthedocs.io/en/stable/snakefiles/remote_files.html#file-transfer-over-ssh-sftp)
functionality which uses [`pysftp`](https://pysftp.readthedocs.io/en/release_0.2.8/pysftp.html#pysftp.Connection).


## Get file list from folder

Once you have some files uploaded to erda, once in a while you could check which files
you already did store there. Assuming you followed the previous setup step, using the
hostname `io.erda.dk`, you can query the files in a `directory` and store the to a file
named `files_and_folders_in_dir.txt`:

`sftp -q io.erda.dk:directory/ <<< "ls" | grep -v '^sftp>' > files_and_folders_in_dir.txt`

### Examples and log

```
sftp -q hela <<<  "ls" | grep -v '^sftp>' > hela_files.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2019_Proteomics/MNT/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2019.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2020_MM/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2020.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2019_Clinical_Proteomics/MNT/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2019_clinical_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2017_Proteomics/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2017_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2017_Clinical_Proteomics/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2017_clinical_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2013_Proteomics/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2013_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2018_Proteomics/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2018_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2018_Clinical_Proteomics/ <<< "ls" | grep -v '^sftp>' > hela_mnt_2018_clinical_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2016_Proteomics/ <<< "ls -S" | grep -v '^sftp>' > hela_mnt_2016_proteomics.txt
sftp -q io.erda.dk:share_hela_raw/MNT_2015_Proteomics/ <<< "ls -S" | grep -v '^sftp>' > hela_mnt_2015_proteomics.txt
```

get a list of all files in the `mq_out` folder on erda (the default folder for storing results):

```
sftp -q io.erda.dk:mq_out/ <<< "ls" | grep -v '^sftp>' > hela_processed.txt
```