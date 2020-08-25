import pandas as pd
import os
from shutil import copyfile
from subprocess import call

### prepare data frame with 3 columns: 
# 'path' is the full path to the .raw file
# 'file' is the .raw file name e.g. 20191210_*.raw
# 'name' is the .raw file name without '.raw' e.g. 20191210_*

# from os.listdir()

DATADIR = '/home/projects/cpr_man/people/s155016/denoms/data/'
SCRIPTDIR = '/home/projects/cpr_man/people/s155016/denoms/script/'

MQ_JOB_TEMPLATE = os.path.join(SCRIPTDIR, 'mq_job_template')

MQ_PARAMETERS = 'mqpar_template.xml'
MQ_PARAMETERS = os.path.join(SCRIPTDIR, MQ_PARAMETERS)

# df = pd.read_csv('my_df.tsv', sep ='\t', header = True)

# loop over all runs
for i in range(df.shape[0]):
    path, file, name = df.iloc[i] #replace by named parameters?
    # create new directory
    os.mkdir(DATADIR+name)
    # copy file  #ToDo: Why copy files?
    copyfile(path, DATADIR + name + '/' + file)
    print('rawfile copied')
    # create mqpar with the correct path and experiment
    mq_parameters_out_file = os.path.join(SCRIPTDIR, 'mqparfiles/mqpar_' + name + '.xml' )
    with open(MQ_PARAMETERS) as infile, open(mq_parameters_out_file, 'w') as outfile:
        for line in infile:
            line = line.replace('PATH', os.path.join(DATADIR, name, file))
            line = line.replace('FILE', file)
            outfile.write(line)
        outfile.close()
        infile.close()
    print('mqpar created')
    # create mq_job file with the correct paths etc.
    mq_job_file = os.path.join(SCRIPTDIR, 'mqjobs/mq_job_' + name)
    with open(MQ_JOB_TEMPLATE) as infile, open(mq_job_file, 'w') as outfile:
        for line in infile:
            line = line.replace('NAME', name)
            line = line.replace('RAWFILE', file)
            outfile.write(line)
        outfile.close()
        infile.close()
    print('mqjob created')
    # run mqjob
    os.chdir(os.path.join(DATADIR, name))
    queue_command = 'qsub '+ mq_job_file
    return_code = call(queue_command, shell=True)
    print('job {} queued out of 528'.format(i+1))
