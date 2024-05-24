#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=cpr_10006 -A cpr_10006
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N  snakemake
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e qsub_logs/${PBS_JOBNAME}.${PBS_JOBID}.e
#PBS -o qsub_logs/${PBS_JOBNAME}.${PBS_JOBID}.o
### Email notification: a=aborts, b=begins, e=ends, n=no notifications
#PBS -m ae -M henry.webel@cpr.ku.dk
### Number of nodes
#PBS -l nodes=1:ppn=2,mem=16gb
### Requesting timeformat is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=1:12:00:00



# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

cd pimms/project # throws an error, but is not consequential.

# Get the values of the parameters from the environment variables
prefix=${prefix:-""}
configfile=${configfile:-""}

# Check if the values are empty
if [ -z "$prefix" ]; then
  echo "Error: Missing required parameters: prefix"
  exit 1
# Check if the values are empty
elif [ -z "$configfile" ]; then
  echo "Error: Missing required parameters: configfile"
  exit 1
else
    echo " # found parameters, see above:"
    echo prefix: $prefix
    echo configfile: $configfile
    echo '####################################################################'
fi

echo folder_experiment $folder_experiment
echo config_split $config_split
echo config_train $config_train

. ~/setup_conda.sh
conda activate vaep

snakemake -s workflow/Snakefile_v2.smk --jobs 10 -k -p -c2 --latency-wait 60 --rerun-incomplete \
--configfile $configfile \
--config frac_mnar=$frac_mnar folder_experiment=$folder_experiment config_split=$config_split config_train=$config_train \
--max-status-checks-per-second 0.1 \
--max-jobs-per-second 1 \
--use-conda \
--default-resources walltime=3600 \
--rerun-trigger mtime \
--cluster "qsub -l walltime={resources.walltime},nodes=1:ppn={threads},mem={resources.mem_mb}mb"\
" -W group_list=cpr_10006 -A cpr_10006 "\
" -e {params.err} -o {params.out}"\
" -N ${prefix}.{params.name}" \
--cluster-status "python workflow/bin/qsub-status_v2.py" &&
echo "done" ||
echo "failed"
