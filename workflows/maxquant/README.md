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
# untrack changes to config
git update-index --assume-unchanged workflows/maxquant/config.yaml 
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

REMOTE_OUT: io.erda.dk
REMOTE_FOLDER: mq_out

# Remote name for fetching files and list of all files
REMOTE_IN: hela
FILES: ../hela_files.txt
FILES_EXCLUDED: log_excluded_files.txt
FILES_FAILED: log_failed.txt
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
> with your MaxQuant version

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

If you don't, snakemake will remind you.
Howwever, snakemake cannot check if the password is correct
before execution, so best verify yourself that it works in the shell you execute.
The `REMOTE` is the same you specified in the `config.yml`:

```bash
sshpass -e sftp -B 258048 REMOTE <<< "pwd"
```

If you set up a SSH connection for your `REMOTE_IN`, you can just set `SSHPASS` to 
anything or comment the two line in the `Snakefile`.

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

#### Using a separate script

```
qsub -V run_sm_on_cluster.sh
```

The `-V` options passes the current environment variables to the shell started by the
run, see [here](http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm)

The script itself contains the cluster execution. Please change the number of parallel jobs
in `run_sm_on_cluster.sh`:

```bash
snakemake --jobs 6 -k --latency-wait 30 --use-envmodules \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 -m f -V "\
"-e {params.logdir} -o {params.logdir}" -n
```

> Once you are sure, remove the dryrun flag `-n`. Dry runs do not necessarily have to be
> sent to the queue.

Alternatively invoked a profile defined from a [template](https://github.com/Snakemake-Profiles/pbs-torque).

Using the profile, the configuration
defined in `config.yaml` and in the `Snakefile` will be used.

```
snakemake --profile pbs-torque --jobs 10 --latency-wait 10 -k
```

#### Logs

All files resulting from executions are stored under the `.snakemake`. See the last file
in the `.snakemake/log` folder for inspecting the process of the currently executed
snakemake job.


## After running snakemake

> The file names can be changed in the `config.yaml`

After snakemake execution of the files in `[hela_files.txt](../hela_files.txt)
you should find new files in the workflow folder [maxquant](vaep/workflows/maxquant):

```
log_completed.txt
log_excluded_files.txt
log_failed.txt
sftp_commands
```

The `log_excluded_files.txt` will be discarded in further workflow runs
(due to being too small) and `log_failed.txt` holds
files which failed although their size is sufficient. The ladder are not automatically
excluded when you re-run snakemake, as the reason for the failure might be on the
server side.

The `sftp_commands` file is the set of commands for batch-mode execution for
transferring files to erda.dk. Assuming the server was reachable when executing the
job, the files should have been transferred during the run. Otherwise you can re-run
the transfer again:

If you set up access to your erda folder appropriatly
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

## Check files on server

In order to see if a corresponding folder on exists on `erda.dk`, you can get a dump of the 
files in the output folder. First get a list of all files in the `mq_out` folder on erda 
(the default folder for storing results, but choose what is in `config.yaml`) :

```
sftp -q io.erda.dk:mq_out/ <<< "ls" | grep -v '^sftp>' > hela_processed.txt
```

> this could be integrated into snakemake _target_ rule.

The `hela_processed.txt` is then the input of the small script `check_current_files.py`:

```
python check_current_files.py -f ../hela_processed.txt -v
```

which dumps the missing, not excluded or failed files into `current_files_to_do.txt`. 
This comparison only checks it the folder for a file exists on the REMOTE if it should be 
completed. `current_files_to_do.txt` can then itself be a new input file or used to remove
some output files. If you are sure set the `forceall` option in snakemake, 
e.g. in `run_sm_on_cluster.sh`.
