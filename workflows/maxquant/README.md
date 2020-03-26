# MaxQuant Workflow

> Provided by Annelaura Bach


In the scripts folder is the `mqpar_template.xml` and the `mq_job_template.sh` and
the `run_mq.py`:

```bash
mqpar_template.xml   # MaxQuant Parameters
mq_job_template.sh   # sumbitted to the queue 
run_mq.py            # script executing MaxQuant
```








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

