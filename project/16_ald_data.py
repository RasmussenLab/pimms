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

logger = vaep.logging.setup_nb_logger()

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100

# %%
folder_data = Path('data/applications/')
folder_data_out = Path('data/single_datasets/')
folder_run = Path('runs/appl_ald_data')
folder_run.mkdir(parents=True, exist_ok=True)

print(*(folder_data.iterdir()), sep='\n')

fnames = dict(
plasma_proteinGroups = folder_data / 'Protein_ALDupgrade_Report.csv',
plasma_aggPeptides = folder_data / 'ald_proteome_spectronaut.tsv',
liver_proteinGroups = folder_data / 'Protein_20200221_121354_20200218_ALD_LiverTissue_PlateS1_Atlaslib_Report.csv',
liver_aggPeptides = folder_data / 'Peptide_20200221_094544_20200218_ALD_LiverTissue_PlateS1_Atlaslib_Report.tsv',
annotations = folder_data / 'ald_experiment_annotations.csv',
clinic = folder_data / 'ald_cli_164.csv',
raw_meta = folder_data / 'ald_metadata_rawfiles.csv')
fnames =vaep.nb.Config.from_dict(fnames) # could be handeled kwargs as in normal dict


# %%
fnames

# %% [markdown]
# ## Parameters

# %%
VAR_PEP = 'PEP.Quantity'
VAR_PG = 'PG.Quantity'

# %% [markdown]
# # Meta data
#
# - sample annotation (to select correct samples)
# - clinical data
# - meta data from raw files (MS machine recorded meta data)

# %% [markdown]
# ## Sample annotations
#
# - `Groups`: more detailed (contains sub-batch information)
# - `Group2`: used to separate samples into cohorts for study
# - `Sample type`: There are liver biopsy samples measured -> select only Plasma samples

# %%
annotations = pd.read_csv(fnames.annotations, index_col='Sample ID')
annotations

# %%
annotations['Participant ID'].value_counts().value_counts() # some only have a blood sample, some both

# %% [markdown]
# ### Select ALD subcohort

# %%
groups = ['ALD']  # 'ALD-validation', 'HP'

# annotations.Groups.value_counts()
annotations.Group2.value_counts()

# %% [markdown]
# ### Select plasma samples

# %%
sel_plasma_samples = (annotations.Group2.isin(['ALD'])) & (annotations['Sample type'] == 'Plasma')
sel_plasma_samples = sel_plasma_samples.loc[sel_plasma_samples].index
annotations.loc[sel_plasma_samples].describe(include=['object', 'string'])

# %% [markdown]
# ### Select liver samples

# %%
groups = ['ALD']  # 'ALD-validation', 'HP'
sel_liver_samples = (annotations.Group2.isin(['ALD'])) & (annotations['Sample type'] == 'Liver')
sel_liver_samples = sel_liver_samples.loc[sel_liver_samples].index
annotations.loc[sel_liver_samples].describe(include=['object', 'string'])

# %% [markdown]
# ## Clinical data

# %%
clinic = pd.read_csv(fnames.clinic, index_col=0)
clinic

# %% [markdown]
# - `idx_overlap_plasma`:  Will be used to select samples with data across datasets available

# %%
print('Missing labels: ', sel_plasma_samples.difference(clinic.index))
idx_overlap_plasma = clinic.index.intersection(sel_plasma_samples)


# %%
clinic.loc[idx_overlap_plasma]

# %% [markdown]
# ## Rawfile information
#
# - [ ] liver samples are currently missing
#

# %%
raw_meta = pd.read_csv(fnames.raw_meta, header=[0, 1], index_col=0)
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
raw_meta = raw_meta.set_index(idx)
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
print("Missing metadata in set of selected labels: ", idx_overlap_plasma.difference(raw_meta.index))
idx_overlap_plasma = idx_overlap_plasma.intersection(raw_meta.index)  # proteomics data has to be part of metadata

# %% [markdown]
# Still save all metadata which is there, but subselect data samples accordingly

# %%
raw_meta.to_csv(folder_data_out / 'raw_meta.csv')

# %% [markdown]
# # Plasma samples
#
# - load samples
# - select based on `sel_plasma_samples`, `raw_meta`  (inclusion in clinical cohort was done before)

# %% [markdown]
# ## Missing samples
#
# From the above we can note that there is
# - no clinical data for `Plate6_F2`
# - no metadata for `Plate2_C1`: re-measured sample which looks fine, but fails with error `"Unable to access the RAW file using the native Thermo library"`
#
# > see section below

# %% [markdown]
# ## (Aggregated) Peptide Data 

# %%
peptides = pd.read_table(fnames.plasma_aggPeptides, low_memory=False)
N_FRIST_META = 8
peptides.shape

# %%
peptides

# %%
peptides.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = peptides.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
peptides = peptides.set_index(list(peptides.columns[:N_FRIST_META])).sort_index(axis=1)

# %%
peptides.loc[:, peptides.columns.str.contains(VAR_PEP)]

# %%
peptides.iloc[:20, :6]

# %% [markdown]
# create new multiindex from column

# %%
sep = '.raw.'
peptides.columns = pd.MultiIndex.from_tuples(peptides.columns.str.split().str[1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
peptides = peptides.stack(0)
peptides

# %% [markdown]
# ### Index meta data

# %% tags=[]
meta = peptides.index.to_frame().reset_index(drop=True)
meta

# %%
meta.describe(include='all')


# %% [markdown]
# ### Select aggregated peptide level data
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
sel_cols = ['Sample ID', 'PEP.StrippedSequence', 'PEP.Quantity'] # selected quantity in last position
sel_data = peptides.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:2])
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
sel_data = sel_data.loc[idx_overlap_plasma]
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

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', feature_name='Aggregated peptides', n_samples=len(sel_data), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_aggPeptides_completness', folder=folder_run)

# %% [markdown]
# ### Select features which are present in at least 25% of the samples

# %%
PROP_FEAT_OVER_SAMPLES = .25
prop = des_data.loc['count'] / len(sel_data)
selected = prop >= PROP_FEAT_OVER_SAMPLES
selected.value_counts()

# %%
sel_data = sel_data.loc[:, selected]
sel_data

# %% [markdown]
# Dump selected data

# %%
fnames.sel_plasma_aggPeptids = folder_data_out / 'ald_plasma_aggPeptides.pkl'
sel_data.to_pickle(fnames.sel_plasma_aggPeptids)

# %% [markdown]
# ## Protein Group data

# %%
pg = pd.read_csv(fnames.plasma_proteinGroups, low_memory=False)
idx_cols = ['PG.ProteinAccessions', 'PG.Genes']
N_FRIST_META = 3
pg

# %%
pg.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = pg.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types # 'PG.Quantity' expected

# %%
pg = pg.set_index(list(pg.columns[:N_FRIST_META])).sort_index(axis=1)
pg.loc[:, pg.columns.str.contains(VAR_PG)]


# %% [markdown]
# Drop index columns which are not selected

# %%
def find_idx_to_drop(df:pd.DataFrame, idx_to_keep:list):
    to_drop = [x for x in pg.index.names if not x in idx_to_keep]
    logger.info("Columnns to drop: {}".format(",".join((str(x) for x in to_drop))))
    return to_drop
    
to_drop = find_idx_to_drop(pg, idx_cols)
pg = pg.reset_index(level=to_drop, drop=True)
pg.head()

# %% [markdown]
# extract long sample name (highly specific to task)
# - whitespace split, taking last position of column name
# - `sep` splits `Sample ID` from `vars`

# %%
sep = '.raw.'
# sep = '.htrms.'
pg.columns = pd.MultiIndex.from_tuples(pg.columns.str.split().str[-1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
pg = pg.stack(0)
pg

# %% [markdown]
# ### Select Protein Group data

# %%
sel_data = pg[[VAR_PG]]
sel_data

# %%
mask = sel_data['PG.Quantity'] == 'Filtered'
print("No. of Filtered entries: ", mask.sum())
sel_data = sel_data.loc[~mask]
sel_data

# %%
sel_data.dtypes

# %%
sel_data = sel_data.squeeze().dropna().astype(float).unstack()
sel_data

# %%
gene_non_unique = sel_data.index.to_frame()["PG.Genes"].value_counts() > 1
gene_non_unique = gene_non_unique[gene_non_unique].index
gene_non_unique

# %%
sel_data.loc[pd.IndexSlice[:, gene_non_unique], :].T.describe()

# %%
sel_data = sel_data.T

idx = sel_data.index.to_series()
idx = idx.str.extract(r'(Plate[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %%
sel_data = sel_data.set_index(idx)
sel_data = sel_data.loc[idx_overlap_plasma]
sel_data

# %%
des_data = sel_data.describe()
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %%
idx_diff = sel_data.index.difference(raw_meta.index)
annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'protein group number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'protein group measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', n_samples=len(sel_data), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_proteinGroups_completness', folder=folder_run)

# %% [markdown]
# ### Select features which are present in at least 25% of the samples

# %%
PROP_FEAT_OVER_SAMPLES = .25
prop = des_data.loc['count'] / len(sel_data)
selected = prop >= PROP_FEAT_OVER_SAMPLES
selected.value_counts()

# %%
sel_data = sel_data.loc[:, selected]
sel_data

# %% [markdown]
# Check for non unique genes after dropping uncommon protein groups.

# %%
gene_non_unique = sel_data.columns.to_frame()["PG.Genes"].value_counts() > 1
gene_non_unique = gene_non_unique[gene_non_unique].index
gene_non_unique

# %% [markdown]
# - less often found -> less intensity on average and on maximum
#
# - [ ] decided if protein group should be subselected
# - alternative selection: per sample, select protein group with highest intensity per sample

# %%
sel_data.T.loc[pd.IndexSlice[:, gene_non_unique], :].T.describe()

# %%
sel_data = sel_data.droplevel(1, axis=1)
sel_data

# %%
sel_data.to_pickle(folder_data_out / 'ald_plasma_proteinGroups.pkl')

# %% [markdown]
# # Liver samples

# %% [markdown]
# ## Peptides

# %%
idx_cols = ['PG.ProteinAccessions', 'PG.Genes', 'Sample ID']
N_FRIST_META = 9
# sel_data = sel_data.reset_index(level=to_remove, drop=True)

# %%
peptides = pd.read_table(fnames.liver_aggPeptides, low_memory=False)
peptides.shape

# %%
peptides

# %%
peptides.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = peptides.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
peptides = peptides.set_index(list(peptides.columns[:N_FRIST_META])).sort_index(axis=1)

# %%
VAR_PEP = 'EG.TotalQuantity'
peptides.loc[:, peptides.columns.str.contains(VAR_PEP)]

# %%
peptides.columns[:10]

# %% [markdown]
# create new multiindex from column

# %%
sep = '.htrms.'
peptides.columns = pd.MultiIndex.from_tuples(peptides.columns.str.split().str[1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
peptides = peptides.stack(0)
peptides

# %% [markdown]
# ### Index meta data

# %% tags=[]
meta = peptides.index.to_frame().reset_index(drop=True)
meta

# %%
meta.describe(include='all')


# %% [markdown]
# ### Select aggregated peptide level data
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
sel_cols = ['Sample ID', 'PEP.StrippedSequence', VAR_PEP] # selected quantity in last position
sel_data = peptides.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:2]).squeeze()
sel_data

# %%
mask = sel_data != 'Filtered'
sel_data = sel_data.loc[mask].astype(float)
sel_data.sort_index()

# %% [markdown]
# Select entry with maximum intensity of `duplicated entries`
#  
# > change of variable and many duplicates -> could be PSM table? (close to evidence?)

# %%
mask_idx_duplicated = sel_data.index.duplicated(False)
sel_data.loc[mask_idx_duplicated].sort_index()

# %%
sel_data = vaep.pandas.select_max_by(df=sel_data.reset_index(), grouping_columns=sel_cols[:-1], selection_column=sel_cols[-1]).set_index(sel_cols[:-1])

# %%
assert sel_data.index.duplicated(False).sum() == 0 , "Still missing values"

# %%
sel_data = sel_data.unstack()
sel_data

# %%
idx = sel_data.index.to_series()
idx = idx.str.extract(r'(PlateS[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %% [markdown]
# - rawfile metadata -> keep 

# %%
sel_data = sel_data.set_index(idx)
sel_data = sel_data.loc[sel_liver_samples]
sel_data

# %%
des_data = sel_data.describe()
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %% [markdown]
# For one raw file no metadata could be extracted (`ERROR: Unable to access the RAW file using the native Thermo library.`)

# %%
# idx_diff = sel_data.index.difference(raw_meta.index)
# annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'peptide number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'peptide measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', feature_name='Aggregated peptides', n_samples=len(sel_data), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_liver_aggPeptides_completness', folder=folder_run)

# %% [markdown]
# ### Select features which are present in at least 25% of the samples

# %%
PROP_FEAT_OVER_SAMPLES = .25
prop = des_data.loc['count'] / len(sel_data)
selected = prop >= PROP_FEAT_OVER_SAMPLES
selected.value_counts()

# %%
sel_data = sel_data.loc[:, selected]
sel_data

# %% [markdown]
# Dump selected data

# %%
fnames.sel_liver_aggPeptids = folder_data_out / 'ald_liver_aggPeptides.pkl'
sel_data.to_pickle(fnames.sel_liver_aggPeptids)

# %% [markdown]
# ## Protein Groups

# %%
df = pd.read_csv(fnames.liver_proteinGroups, low_memory=False)
idx_cols = ['PG.ProteinAccessions', 'PG.Genes']
N_FRIST_META = 5
df

# %%
# find_idx_to_drop(df, idx_cols)


# %%
df.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = df.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types # 'PG.Quantity' expected

# %%
df = df.set_index(list(df.columns[:N_FRIST_META])).sort_index(axis=1)
df.loc[:, df.columns.str.contains(VAR_PG)]


# %% [markdown]
# Drop index columns which are not selected

# %%
# to_drop = find_idx_to_drop(df, idx_cols)
# df = df.reset_index(level=to_drop, drop=True)

# %% [markdown]
# extract long sample name (highly specific to task)
# - whitespace split, taking last position of column name
# - `sep` splits `Sample ID` from `vars`

# %%
sep = '.htrms.'
df.columns = pd.MultiIndex.from_tuples(df.columns.str.split().str[-1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
df = df.stack(0)
df

# %% [markdown]
# ### Select Protein Group data

# %%
df = df[[VAR_PG]]
df

# %%
mask = df['PG.Quantity'] == 'Filtered'
print("No. of Filtered entries: ", mask.sum())
df = df.loc[~mask]
df

# %%
sel_cols = ['PG.ProteinAccessions', 'PG.Genes', 'Sample ID', VAR_PG] # last one gives quantity
df = df.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:-1])

# %%
df.dtypes

# %%
df = df.squeeze().dropna().astype(float).unstack()
df

# %%
gene_non_unique = df.index.to_frame()["PG.Genes"].value_counts() > 1
gene_non_unique = gene_non_unique[gene_non_unique].index
gene_non_unique

# %%
df.loc[pd.IndexSlice[:, gene_non_unique], :].T.describe()

# %%
df = df.T

idx = df.index.to_series()
idx = idx.str.extract(r'(PlateS[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %%
df = df.set_index(idx)
# df = df.loc[idx_overlap_liver] # missing
df = df.loc[sel_liver_samples]
df

# %%
des_data = df.describe()
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap
# - no raw data yet

# %%
idx_diff = df.index.difference(raw_meta.index)
annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'protein group number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'protein group measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', n_samples=len(df), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
fnames.fig_liver_pg_completness = folder_run / 'data_liver_proteinGroups_completness'
vaep.savefig(fig, name=fnames.fig_liver_pg_completness)

# %% [markdown]
# ### Select features which are present in at least 25% of the samples

# %%
PROP_FEAT_OVER_SAMPLES = .25
prop = des_data.loc['count'] / len(df)
selected = prop >= PROP_FEAT_OVER_SAMPLES
selected.value_counts()

# %%
df = df.loc[:, selected]
df

# %% [markdown]
# Check for non unique genes after dropping uncommon protein groups.

# %%
gene_non_unique = df.columns.to_frame()["PG.Genes"].value_counts() > 1
gene_non_unique = gene_non_unique[gene_non_unique].index
gene_non_unique

# %% [markdown]
# - less often found -> less intensity on average and on maximum
#
# - [ ] decided if protein group should be subselected
# - alternative selection: per sample, select protein group with highest intensity per sample

# %%
df.T.loc[pd.IndexSlice[:, gene_non_unique], :].T.describe()

# %%
df = df.droplevel(1, axis=1)
df

# %%
fnames.sel_liver_proteinGroups = folder_data_out / 'ald_liver_proteinGroups.pkl'
df.to_pickle(fnames.sel_liver_proteinGroups)

# %% [markdown]
# # All file names
# - inputs
# - output data (`pkl`)
# - figures

# %%
fnames

# %%
cfg_dump = 'config/ald_data.yaml'
fnames.dump(cfg_dump)
