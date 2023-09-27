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
#PBS -l nodes=1:ppn=1,mem=8gb
### Requesting timeformat is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=1:12:00:00
### Forward all environment variables



# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

. ~/setup_conda.sh
conda activate vaep

snakemake --jobs 10 -k -p --latency-wait 60 --rerun-incomplete \
--default-resources walltime=3600 \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 -V"\
" -e {params.err} -o {params.out}" \
--cluster-status "python ../workflows/maxquant/qsub-status.py" &&
echo "done" ||
echo "failed"
