# MaxQuant Workflow
```
module load tools
module load anaconda3/2019.10
conda init bash
bash
conda create -c conda-forge -c bioconda -n snakemake snakemake=5.3 
#  ~/.conda/envs/snakemake
conda activate snakemake
git update-index --assume-unchanged workflows/maxquant/config.yaml # untrack changes to config
```
## Snakemake

1. create conda environment for snakemake
    ```
    conda env create -f vaep/workflows/maxquant/environment.yml 
    ```
2. create cluster profile using [cookiecutter](https://github.com/Snakemake-Profiles/pbs-torque)
    ```
    mkdir -p ~/.config/snakemake
    cd ~/.config/snakemake
    cookiecutter https://github.com/Snakemake-Profiles/pbs-torque.git
    ```

## Example `config.yaml` for Workflow
Here the username is `henweb` and the group is Simon Rasmussen's group `cpr_10006`.

```
DATADIR: /home/projects/cpr_10006/people/henweb/hela/
SCRIPTDIR: /home/projects/cpr_10006/people/henweb/vaep/workflows/maxquant/

#Either your own or 
MAXQUANTEXE: /home/projects/cpr_10006/people/henweb/MaxQuant_1.6.12.0/MaxQuant/bin/MaxQuantCmd.exe
# MAXQUANTEXE: /services/tools/maxquant/1.6.7.0/MaxQuant.exe

MQ_PAR_TEMP: /home/projects/cpr_10006/people/henweb/vaep/workflows/maxquant/mqpar_template_1.6.xml
THREATS_MQ: 8
```

> You have to specify the fasta file paths manually in the parameter template file
> referenced in MQ_PAR_TEMP, e.g. `/home/projects/cpr_10006/people/henweb/fasta/myfasta.fasta`

If you specify passwords in your config file you might want to restrict permissions to your user
```
chmod 600 config.yaml
```

### Load MaxQuant
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

[qsub](http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm)

```bash
#snakemake --jobs 2 --latency-wait 30 --cluster "qsub -W group_list=cpr_10006 -A cpr_10006 -m f"
snakemake --jobs 2 -k --latency-wait 30 --use-envmodules \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 -m f -V "\
"-e log_qsub/snakejob.{rule}.{wildcards.file}.e$PBS_JOBID -o log_qsub/snakejob.{rule}.{wildcards.file}.o$PBS_JOBID"
```

```bash
#snakemake --jobs 2 --latency-wait 30 --cluster "qsub -W group_list=cpr_10006 -A cpr_10006 -m f" -n
snakemake --jobs 2 -k --latency-wait 30 --use-envmodules \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 -m f -V "\
"-e {params.logdir} -o {params.logdir}" -n
```

Alternatively invoked a profile defined from the template before. 

Using the profile defined previously, the configuration 
defined in `config.yaml` and in the `Snakefile` will be used.

```
snakemake --profile pbs-torque --jobs 10 --latency-wait 10 -k 
```

## Python Template
> Provided by Annelaura Bach


In the scripts folder is the `mqpar_template.xml` and the `mq_job_template.sh` and
the `run_mq.py`:

```bash
mqpar_template.xml   # MaxQuant Paramete

mq_job_template.sh   # sumbitted to the queue 
run_mq.py            # script executing MaxQuant
```

## Snakemake
Snakemake is a framework for execution of workflows on UNIX based systems.
It is written in the line of thought of 
[`GNU Makefiles`](https://www.opensourceforu.com/2012/06/gnu-make-in-detail-for-beginners/),
but as an extension to Python rather than `C/C++`.

### Setup
```
conda install -n snakemake snakemake pygraphviz
```

## Transfer results to erda

find ./hela/  -name '*txt*' -type d -print
find ./hela/  -path ./*/combined/txt -type d


## Typical directory structure
Having a working directory called `dir`
```
dir/MaxQuant  
dir/data  
dir/result  
dir/script
```

You have to have a reference fasta file: `dir/data/fasta/ref_proteome.fasta` (UNIPROT)
Remember to change the filename in the `mqpar_template.xml`.


You need to edit the fasta file in the mqpar file and probably a lot more, if you want match between runs and other functions
You need to edit the mq_job template with your computerome configurations 
You need to prepare a dataframe and change file paths in the run_mq.py file

