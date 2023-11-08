# Scripts for pbs-torque cluster execution

We ran the software partly on a pbs-torque cluster

`qsub-status_v2.py` is used by snakemake to  query the status of a submitted job in case the job,
in case the job is not ran locally within the main process running snakemake.

`create_qsub_commands.py` is a script which create some job submission commands.

`jobscript.sh` is a script which sets up conda before the subcommand create from within
a snakemake job is run.


> None of this is needed in case snakemake is non-distributed on a single node.