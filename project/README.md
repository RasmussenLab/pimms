# Paper project


notebook  | Description
---  |  --- 
00_maxquant_file_processing.ipynb | Notebook for processing files in FTP environement (erda)
01_FASTA_tryptic_digest.ipynb     | Analyze fasta file used for peptide identification
1_maxquant_file_analysing.ipynb   | 
1_data_exploration_peptides.ipynb | 
1_data_exploration_proteins.ipynb | 
10_training_data.ipynb            | ERDA: Build training data dump
2_clustering_proteins.ipynb       | 
3_select_data.ipynb               | Visualize all data (and try running models) <br> 16GB RAM needed
4_fine_tune_model.ipynb           | Select a protein data dump and play with analysis
id_mapper.ipynb                   | 
VAEP_POC.ipynb                    | Presentation for POC

explore_data.ipynb                | Load an MQ txt output folder and browse data

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
