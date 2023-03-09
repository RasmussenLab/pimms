# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import json
from pathlib import Path

import pandas as pd

# %%
from config import FN_FASTA_DB
from config import FOLDER_PROCESSED, FOLDER_DATA, FOLDER_TRAINING

class Data:
    pass
data = Data()

# %%
data.sample_folders = [folder for folder in FOLDER_PROCESSED.iterdir() if folder.is_dir() ]
data.sample_folders[-10:]

# %%
data.data_completeness = {}
for folder in data.sample_folders:
    key = folder.stem
    try:
        d = (folder / '0_completeness_all_genes.json').read_text()
        d = json.loads(d)
        data.data_completeness[key] = d
    except FileNotFoundError:
        print(f"skip: {key}")
data.data_completeness = pd.DataFrame.from_dict(data.data_completeness, orient='index')

# %%
data.samples_per_gene = data.data_completeness.count().sort_values(ascending=False)
data.samples_per_gene[:10]

# %%
gene = data.samples_per_gene.index[1]
gene


# %%
class Analysis:
    
    def __init__(self, gene):
        self.gene = gene

analysis_gene = Analysis(gene)

# %%
setattr(data , gene, {})
current_dict = getattr(data, gene)
for folder in data.sample_folders:
    key = folder.stem
    try:
        d = (folder / f"{gene}.json").read_text()
        d = json.loads(d)
        current_dict[key] = d['Intensity']
    except FileNotFoundError:
        print(f"No dump of {gene} for {key}")

setattr(analysis_gene, 'data', pd.DataFrame.from_dict(current_dict, orient='index'))

# %%
analysis_gene.data.count().sort_values()

# %% [markdown]
# ## Select subset of peptides
#
# - peptides with few measurements might be non-unique (filter non-gene unqiue peptides?!)
# - aggregate peptides in a future step using
#     - fill NA values of well-cleave peptides based on miscleavages?
#     - discard non-often measured peptides?

# %%
c = analysis_gene.data.count().sort_index()
c /= c.max()
c.plot(kind='hist')

# %%
analysis_gene.peptides = c[c > 0.6 ].index
len(analysis_gene.peptides)

# %%
analysis_gene.features_per_sample = analysis_gene.data[analysis_gene.peptides].notna().sum(axis=1)

analysis_gene.features_per_sample.hist()

# %%
analysis_gene.peptides_per_run = analysis_gene.data.notna().sum(axis=1)

_ = analysis_gene.peptides_per_run.plot(kind='hist', bins=10)
analysis_gene.peptides_per_run.sort_values()

# %%
with open(FOLDER_TRAINING / f'{gene}.json', 'w') as f:
    analysis_gene.data.loc[analysis_gene.peptides_per_run > 11].to_json(f)

# %%
analysis_gene.data.loc[analysis_gene.peptides_per_run > 11]
