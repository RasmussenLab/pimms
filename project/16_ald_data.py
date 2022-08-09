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
import vaep

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100

# %%
folder_data = Path('data/applications/')
folder_data_out = Path('data/single_datasets/')
folder_run = Path('runs/ald_study')
folder_run.mkdir(parents=True, exist_ok=True)

print(*(folder_data.iterdir()), sep='\n')

f_proteome = folder_data / 'ald_proteome_spectronaut.tsv'
f_annotations = folder_data / 'ald_experiment_annotations.csv'
f_clinic = folder_data / 'ald_cli_164.csv'
f_raw_meta = folder_data / 'ald_metadata_rawfiles.csv'

# %%
data = pd.read_table(f_proteome, low_memory=False)
data.shape

# %%
data

# %%
data.iloc[:, :8].describe(include='all')

# %%
column_types = data.iloc[:, 8:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
data = data.set_index(list(data.columns[:8])).sort_index(axis=1)

# %%
data.loc[:, data.columns.str.contains(column_types[0])]

# %%
data.iloc[:20, :6]

# %% [markdown]
# create new multiindex from column

# %%
data.columns = pd.MultiIndex.from_tuples(data.columns.str.split().str[1].str.split(
    '.raw.').to_series().apply(tuple), names=['Sample ID', 'vars'])
data = data.stack(0)
data

# %% [markdown]
# ## Meta data
#
# - sample annotation (to select correct samples)
# - meta data from Spectronaut ouput
# - clinical data
# - meta data from raw files (MS machine recorded meta data)

# %% [markdown]
# ### From Spectronaut file

# %%
meta = data.index.to_frame().reset_index(drop=True)
meta

# %%
meta.describe()

# %% [markdown]
# ### Sample annotations
#
# - `Groups`: more detailed (contains sub-batch information)
# - `Group2`: used to separate samples into cohorts for study
# - `Sample type`: There are liver biopsy samples measured -> select only Plasma samples

# %%
annotations = pd.read_csv(f_annotations, index_col='Sample ID')
annotations

# %% [markdown]
# Select ALD subcohort

# %%
# annotations.Groups.value_counts()
annotations.Group2.value_counts()

# %%
groups = ['ALD']  # 'ALD-validation', 'HP'
selected = (annotations.Group2.isin(['ALD'])) & (annotations['Sample type'] == 'Plasma')
selected = selected.loc[selected].index
annotations.loc[selected].describe(include=['object', 'string'])

# %% [markdown]
# ### Clinical data

# %%
clinic = pd.read_csv(f_clinic, index_col=0)
clinic

# %% [markdown]
# - `idx_overlap`:  Will be used to select samples with data across datasets available

# %%
print('Missing labels: ', selected.difference(clinic.index))
idx_overlap = clinic.index.intersection(selected)


# %%
clinic.loc[idx_overlap]

# %% [markdown]
# ### Rawfile information

# %%
raw_meta = pd.read_csv(f_raw_meta, header=[0, 1], index_col=0)
raw_meta.index.name = "Sample ID (long)"
raw_meta

# %% [markdown]
# Measurements are super homogenous

# %%
raw_meta.describe()

# %%
idx = raw_meta.index.to_series()
idx = idx.str.extract(r'(Plate[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %%
raw_meta = raw_meta.reset_index().set_index(idx)
raw_meta

# %%
df_meta_rawfiles_columns = raw_meta.columns  # needs to go to Config which is not overwriteable by attribute selection
meta_raw_names = raw_meta.columns.droplevel()
assert meta_raw_names.is_unique
meta_raw_names.name = None
raw_meta.columns = meta_raw_names

# %%
raw_meta.loc[['Plate6_F2']]

# %%
print("Missing metadata in set of selected labels: ", idx_overlap.difference(raw_meta.index))
idx_overlap = idx_overlap.intersection(raw_meta.index)  # proteomics data has to be part of metadata

# %% [markdown]
# Still save all metadata which is there, but subselect data samples accordingly

# %%
raw_meta.to_pickle(folder_data_out / 'raw_meta.pkl')

# %% [markdown]
# ## Missing samples
#
# From the above we can note that there is
# - no clinical data for `Plate6_F2`
# - no metadata for `Plate2_C1`
#
# > see section below

# %% [markdown]
# ## Select Proteomics data
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
sel_data = sel_data.squeeze().dropna().astype(float).unstack()
sel_data

# %%
idx = sel_data.index.to_series()
idx = idx.str.extract(r'(Plate[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %% [markdown]
# - rawfile metadata -> keep 

# %%
sel_data = sel_data.set_index(idx)
sel_data = sel_data.loc[idx_overlap]
sel_data

# %%
des_data = sel_data.describe()
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %% [markdown]
# For one raw file no metadata could be extracted (`ERROR: Unable to access the RAW file using the native Thermo library.`)

# %%
idx_diff = sel_data.index.difference(raw_meta.index)
annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'peptide number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'peptide measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(), feat_col_name='count', feature_name='Aggregated peptides', n_samples=len(sel_data), ax=None, **kwargs)

# %% [markdown]
# Dump selected data

# %%
sel_data.to_pickle(folder_data_out / 'ald_aggPeptides_spectronaut.pkl')
