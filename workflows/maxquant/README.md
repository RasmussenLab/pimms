# MaxQuant Workflow

## Setup on CR2 
If you already have a working version of snakemake, consider to skip this:
```
module load tools
module load anaconda3/2019.10
conda init bash
bash
conda env create -f vaep/workflows/maxquant/environment.yml 
#  ~/.conda/envs/snakemake
conda activate snakemake
git update-index --assume-unchanged workflows/maxquant/config.yaml # untrack changes to config
```

## Example `config.yaml` for Workflow
Here the username is `henweb` and the group is Simon Rasmussen's group `cpr_10006`.

```
DATADIR: /home/projects/cpr_10006/people/henweb/hela/
SCRIPTDIR: /home/projects/cpr_10006/people/henweb/vaep/workflows/maxquant/

#Either your own or (see below)
MAXQUANTEXE: /home/projects/cpr_10006/people/henweb/MaxQuant_1.6.12.0/MaxQuant/bin/MaxQuantCmd.exe
# MAXQUANTEXE: /services/tools/maxquant/1.6.7.0/MaxQuant.exe

MQ_PAR_TEMP: /home/projects/cpr_10006/people/henweb/vaep/workflows/maxquant/mqpar_template_1.6.xml
THREATS_MQ: 8

# Remote name for fetching files and list of all files
REMOTE: hela  
FILES: ../hela_files.txt
```

> You have to specify the fasta file paths manually in the parameter template file
> referenced in MQ_PAR_TEMP, e.g. `/home/projects/cpr_10006/people/henweb/fasta/myfasta.fasta`

If you specify passwords in your config file you might want to restrict permissions to your user
```
chmod 600 config.yaml
```

### Find MaxQuant exectuable
You can either use a pre-exisiting MaxQuant installation or a  new one.
Once you know the path, you do not need to load the module explicitly 
into your set of environment variables.
```
module load mono/5.20.1.19 maxquant/1.6.7.0
export | grep MAXQUANT # find path to MaxQuant executable
```

> It seems that also on minor version updates the parameter file of MaxQuant is
> not preserved. Make sure that your template parameter file is working together
> with your MaxQuant version (by checking that locally?)

## Test your Workflow - Dry-Run of Snakemake

Make sure to be in the MaxQuant workflow folder `workflows/maxquant/` and 
have a session which you can reconnect to (using e.g. `screen` or `tmux`).

### Set `SSHPASS`
This workflow uses in the current implementation a password protected sftp 
connection. In order to login your local environment in which you run 
snakemake has to have a password,  `<PASSWORD>` set.

```bash
export SSHPASS=<PASSWORD>
```
If you don't snakemake will remind you. 
Howwever, snakemake cannot check if the password is correct 
before execution, so best verify yourself that it works in the shell you execute.
The `REMOTE` is the same you specified in the `config.yml`:

```bash
sshpass -e sftp -B 258048 REMOTE <<< "pwd" 
```

### Dry-RUN

```
snakemake -n
snakemake -n --report
```

### Run locally

Either on your computer or in an interactive shell (`iqsub`)

Running snakemake with many repeated sample which might fail, you can type:
```
snakemake -k
```

### Run on cluster

```
qsub -V run_sm_on_cluster.sh 
```

####
[qsub](http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm)

```bash
snakemake --jobs 10 -k --latency-wait 30 --use-envmodules \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 -m f -V "\
"-e {params.logdir} -o {params.logdir}" -n
```
> Once you are sure, remote the dryrun flag `-n`

Alternatively invoked a profile defined from the template before. 

Using the profile defined previously, the configuration 
defined in `config.yaml` and in the `Snakefile` will be used.

```
snakemake --profile pbs-torque --jobs 10 --latency-wait 10 -k 
```

## Transfer data to erda.dk

After snakemake execution of the files in `[hela_files.txt](../hela_files.txt)
you should find three new files in the workflow folder [maxquant](vaep/workflows/maxquant):

```
completed.txt
excluded_files.txt
failed.txt
sftp_commands
```

The `excluded_files.txt` will be discarded in further workflow runs 
(due to being too small) and `failed.txt` holds 
files which failed although their size is sufficient.

The `sftp_commands` file is the set of commands for batch-mode execution for 
transferring files to erda.dk. If you set up access to your erda folder appropriatly
you should be able to connect to `erda <your-hostname>`. I named it `erda io.erda.dk`.
If you can connect using this command, execute the sftp command in batch mode providing
`sftp_commands` as an argument in order to store the files in a `hela` folder on your
erda root directory.

```
sftp -B 258048 -b sftp_commands io.erda.dk
```

> afterwards one should rename the sftp_commands file or move it to an archive folder.

## Find output files
> Look up only for now
Find MQ output files in `hela` folder and remove them by age:
```
find ./hela/  -name '*txt*' -type d -print
find ./hela/  -path ./*/combined/txt -type d
ls ./hela/ -ltr # check for old files
find ./hela/ -mtime +2 
#find ./hela/ -mtime +2 -exec rm {} \;
#find ./hela/ -mtime +2 -exec rmdir {} \; -type d
#find ./hela/ -type d -empty -delete
#find ./hela/ -mtime +2 -exec rm -r {} \; 
```