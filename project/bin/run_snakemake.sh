#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=cpr_10006 -A cpr_10006
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N  sn_grid
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e qsub_logs/${PBS_JOBNAME}.${PBS_JOBID}.e
#PBS -o qsub_logs/${PBS_JOBNAME}.${PBS_JOBID}.o
### Email notification: a=aborts, b=begins, e=ends, n=no notifications
#PBS -m ae -M henry.webel@cpr.ku.dk
### Number of nodes
### other: #PBS -l nodes=1:ppn=20:mem:40g
#PBS -l nodes=1:ppn=40
### Requesting timeformat is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=1:00:00:00
### Forward all environment variables
### if authentification is done using pw in the environment
#PBS -V


# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# start_conda
. ~/setup_conda.sh
conda activate vaep

# try to influence how many jobs are run in parallel in one job training a model
export MKL_NUM_THREADS=5 

snakemake --snakefile workflow/Snakefile_grid.smk --rerun-incomplete -f -j 4 -c 20

