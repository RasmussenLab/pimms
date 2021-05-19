# Paper project

## Erda Notebooks
notebook  | Description
---  |  --- 
00_mq_aggregate_summaries.ipynb   | Aggregate current summary files from MQ runs
00_mq_count_peptides.ipynb        | Aggregate peptide information from all MQ peptides.txt files
01_explore_raw_MQ_data.ipynb      | Load an MQ txt output folder and browse data
01_FASTA_data_agg_by_gene.ipynb   | Investigate possibility to join proteins by gene
01_FASTA_tryptic_digest.ipynb     | Analyze fasta file used for peptide identification
04_all_raw_files.ipynb            | Find duplicate raw files, analyze sizes
1_data_exploration_peptides.ipynb | Describe current peptides training data
1_data_exploration_proteins.ipynb | \[NEEDS UPDATE\] Describe protein training data 
1_maxquant_file_analysing.ipynb   | \[NEEDS UPDATE\] how many peptides are available
10_training_data.ipynb            | ERDA: Build training data dump
11_training_data_exploration_peptides.ipynb | Analyze dump of training data for patterns<br> Visualize key metrics
12_experiment_01_small_example |  Assess inital model performance on smaller training data<br> - linear vs log transformed data<br> - vary number of layers and neurons in layers<br> - compare performance in original space
2_clustering_proteins.ipynb       | \[NEEDS UPDATE\] PCA protein analysis from Annelaura w/ initial data
3_select_data.ipynb               | Visualize all data (and try running models) <br> 16GB RAM needed
3_select_training_data.ipynb      | \[NEEDS UPDATE\] Sort training data by gene
5_0_summaries                     | Analyzse summaries.txt data from all samples
id_mapper.ipynb                   | train models per gene, see overlaps in proteins, see coverage of proteins with observed peptides, align overlapping peptide sequences
json_formats.ipynb                | Investigate storring training data as json with correct encoding
VAEP_01_MaxQuantOutput.ipynb      | Show MaxQuantOutput class behaviour
VAEP_POC.ipynb                    | Presentation for POC

> exectued notebook version are stored under [`doc/ipynbs`](doc/ipynbs)  exported a

## Explore Dataset

The data set can be explored in a streamlit app by running locally:
```
streamlit run bin/streamlit_dashboard.py
```

In order to develop new functionality, one possibility is to run the
script in an interactive ipython session:
```
ipython -i bin/streamlit_dashboard.py
``` 
