# MaxQuant Workflow

## Python Template
> Provided by Annelaura Bach


In the scripts folder is the `mqpar_template.xml` and the `mq_job_template.sh` and
the `run_mq.py`:

```bash
mqpar_template.xml   # MaxQuant Parameters
mq_job_template.sh   # sumbitted to the queue 
run_mq.py            # script executing MaxQuant
```

## Snakemake
Snakemake is a framework for execution of workflows on UNIX based systems.
It is written in the line of thought of 
[`GNU Makefiles`](https://www.opensourceforu.com/2012/06/gnu-make-in-detail-for-beginners/),
but as an extension to Python rather than `C/C++`.

### Setup
```
conda install -n snakemake snakemake pygraphviz
```


### Dry-RUN

```
snakemake -n
snakemake -n --report
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

