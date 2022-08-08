#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=cpr_10006 -A cpr_10006
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N  snakemake
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e ${PBS_JOBNAME}.${PBS_JOBID}.e
#PBS -o ${PBS_JOBNAME}.${PBS_JOBID}.o
### Email notification: a=aborts, b=begins, e=ends, n=no notifications
#PBS -m ae -M henry.webel@cpr.ku.dk
### Number of nodes
### other: #PBS -l nodes=1:ppn=40:gpus=1
#PBS -l nodes=1:ppn=40
### Requesting timeformat is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=1:00:00:00
### Forward all environment variables
### if authentification is done using pw in the environment
#PBS -V

module load tools git/2.15.0
module load anaconda3/2021.11

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/services/tools/anaconda3/2021.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/services/tools/anaconda3/2021.11/etc/profile.d/conda.sh" ]; then
        . "/services/tools/anaconda3/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/services/tools/anaconda3/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR


conda activate vaep

# try to influence how many jobs are run in parallel in one job training a model
export MKL_NUM_THREADS=10 

snakemake --snakefile workflow/Snakefile_grid.smk --rerun-incomplete -f -j 4 -c 40

