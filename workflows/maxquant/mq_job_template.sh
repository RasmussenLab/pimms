#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=cpr_man -A cpr_man
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N NAME 
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e NAME.err
#PBS -o NAME.log
### Only send mail when job is aborted or terminates abnormally
#PBS -M annelaura.bach@cpr.ku.dk
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=28,mem=100gb
### Requesting time - 12 hours - overwrites **long** queue setting
#PBS -l walltime=100:00:00

# Go to the directory from where the job was submitted (initial directory is $HOME)
module load tools
module load mono/6.0.0.327

mono --version
mono /home/projects/cpr_man/people/s155016/denoms/MaxQuant/bin/MaxQuantCmd.exe /home/projects/cpr_man/people/s155016/denoms/script/mqparfiles/mqpar_NAME.xml

rm /home/projects/cpr_man/people/s155016/denoms/data/NAME/RAWFILE
echo "rawfile deleted"

mv /home/projects/cpr_man/people/s155016/denoms/data/NAME/combined/txt/proteinGroups.txt /home/projects/cpr_man/people/s155016/denoms/result/proteinGroups/NAME_proteinGroups.txt
mv /home/projects/cpr_man/people/s155016/denoms/data/NAME/combined/txt/peptides.txt /home/projects/cpr_man/people/s155016/denoms/result/peptides/NAME_peptides.txt
echo "result files moved"

echo "Done"
