# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # ALD Study

# %%
from pathlib import Path
import pandas as pd

# %%
file = 'data/applications/ald_proteome_spectronaut.csv'

# %%
data = pd.read_csv(file, delimiter=';', low_memory=False)
data.shape

# %%
data

# %%
data.iloc[:,:8].describe()

# %%
column_types = data.iloc[:,8:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe() ## .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
data = data.set_index(list(data.columns[:8])).sort_index(axis=1)

# %%
data.loc[:,data.columns.str.contains(column_types[0])]

# %%
data.iloc[:20,:6]

# %% [markdown]
# create new multiindex from column

# %%
data.columns = pd.MultiIndex.from_tuples(data.columns.str.split().str[1].str.split('.raw.').to_series().apply(tuple), names = ['Sample ID', 'vars'])
data = data.stack(0)
data

# %% [markdown]
# ## Meta data

# %%
meta = data.index.to_frame().reset_index(drop=True)
meta

# %%
meta.describe()

# %% [markdown]
# ## Data and metadata
#
# taken from [Spectronaut manuel](https://biognosys.com/resources/spectronaut-manual/)
#
# feature | description 
# --- | ---
# PEP.IsProteinGroupSpecific | True or False. Tells you whether the peptide only belongs to one Protein Group.
# PEP.StrippedSequence | -
# PEP.IsProteotypic |  -
# PEP.PeptidePosition | -
# PG.Cscore | - 
# PG.ProteinAccessions | -
# PG.Genes | - 
# PEP.Quantity | The quantitative value for that peptide as defined in the settings.
# EG.PrecursorId | Unique Id for the precursor: [modified sequence] plus [charge] 
# EG.Qvalue | The q-value (FDR) of the EG.
# EG.TotalQuantity (Settings) | The quantitative value for that EG as defined in the settings. 
#
# > Headers related to Peptides (PEP) as defined in the settings. Many headers related to Peptides are self-explanatory. Here are the most relevant and some which are not too obvious. 
#
# > Headers related to Peptides (PEP) as defined in the settings. Many headers related to Peptides are self-explanatory. Here are the most relevant and some which are not too obvious. 
#
# After discussing with Lili, `PEP.Quantity` is the fitting entity for each unique aggregated Peptide. Duplicated entries are just to drop

# %%
sel_cols = ['Sample ID', 'PEP.StrippedSequence', 'PEP.Quantity']
sel_data = data.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:2])
sel_data

# %%
sel_data.squeeze().dropna().unstack()

# %%
sel_data.to_pickle('data/single_datasets/ald_aggPeptides_spectronaut.pkl')

# %%
