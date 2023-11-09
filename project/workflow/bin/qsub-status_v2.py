#!/usr/bin/env python3
# https://github.com/Snakemake-Profiles/pbs-torque/blob/master/%7B%7Bcookiecutter.profile_name%7D%7D/pbs-status.py
import sys
import subprocess

jobid = sys.argv[1]

try:
    # ! do net query full information
    qstat = subprocess.run("qstat {}".format(jobid), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    res = qstat.stdout.decode(errors='ignore')

    if "C" in res:
        full = subprocess.run("qstat -f -x {}".format(jobid), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        full = full.stdout.decode(errors='ignore')
        if "<exit_status>0</exit_status>" in full:
            print("success")
        else:
            print("failed")
    else:
        print("running")

except (subprocess.CalledProcessError, IndexError, KeyboardInterrupt) as e:
    print("failed")