#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=cpr_10006 -A cpr_10006
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N  
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e ${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -o ${PBS_JOBNAME}.o${PBS_JOBID}
### Email notification: a=aborts, b=begins, e=ends, n=no notifications
#PBS -m an -M henry.webel@cpr.ku.dk
### Number of nodes
#PBS -l nodes=1:ppn=20,mem=40gb
### Requesting timeformat is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=12:00:00

# Go to the directory from where the job was submitted (initial directory is $HOME)
module load tools
module load mono/6.8.0.105
module load maxquant/1.6.7.0


mono --version
#mono /home/projects/cpr_man/people/s155016/denoms/MaxQuant/bin/MaxQuantCmd.exe /home/projects/cpr_man/people/s155016/denoms/script/mqparfiles/mqpar_NAME.xml

rm /home/projects/cpr_man/people/s155016/denoms/data/NAME/RAWFILE
echo "rawfile deleted"

mv /home/projects/cpr_man/people/s155016/denoms/data/NAME/combined/txt/proteinGroups.txt /home/projects/cpr_man/people/s155016/denoms/result/proteinGroups/NAME_proteinGroups.txt
mv /home/projects/cpr_man/people/s155016/denoms/data/NAME/combined/txt/peptides.txt /home/projects/cpr_man/people/s155016/denoms/result/peptides/NAME_peptides.txt
echo "result files moved"

echo "Done"
