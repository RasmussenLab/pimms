# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ALD Study

# %%
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import vaep

logger = vaep.logging.setup_nb_logger()

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100

# %%
folder_data = Path('data/ALD_study/')
folder_data_out = folder_data / 'processed'
folder_data_out.mkdir(parents=True, exist_ok=True)
folder_run = Path('runs/appl_ald_data')
folder_run.mkdir(parents=True, exist_ok=True)

print(*(folder_data.iterdir()), sep='\n')

fnames = dict(
    plasma_proteinGroups=folder_data / 'Protein_ALDupgrade_Report.csv',
    plasma_aggPeptides=folder_data / 'ald_proteome_spectronaut.tsv',
    liver_proteinGroups=folder_data / 'Protein_20200221_121354_20200218_ALD_LiverTissue_PlateS1_Atlaslib_Report.csv',
    liver_aggPeptides=folder_data / 'Peptide_20220819_100847_20200218_ALD_LiverTissue_PlateS1_Atlaslib_Report.csv',
    annotations=folder_data / 'ald_experiment_annotations.csv',
    clinic=folder_data / 'labtest_integrated_numeric.csv',
    raw_meta=folder_data / 'ald_metadata_rawfiles.csv')
fnames = vaep.nb.Config.from_dict(fnames)  # could be handeled kwargs as in normal dict


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
annotations['Participant ID'].value_counts().value_counts()  # some only have a blood sample, some both

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

# %%
idx_qc_plasma = annotations.Group2[annotations.Group2 == 'QC'].index
with (folder_data_out / 'qc_samples.yaml').open('w') as f:
    yaml.safe_dump(idx_qc_plasma.to_list(), f)
idx_qc_plasma

# %%
idx_qc_liver = annotations.Group2[annotations.Group2 == 'QC_liver'].index
with (folder_data_out / 'qc_samples.yaml').open('w') as f:
    yaml.safe_dump(idx_qc_liver.to_list(), f)
idx_qc_liver

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

# %%
clinic['abstinent_num'] = (clinic["currentalc"] == 0.00).astype(int)
clinic[['abstinent_num', 'currentalc']].describe()

# %% [markdown]
# Kleiner score of 0.5 was assigned as value of 0-1 without biopsy. Is set to NA.

# %%
clinic["kleiner"] = clinic["kleiner"].replace({-1: np.nan, 0.5: np.nan})
clinic["kleiner"].value_counts()

# %%
clinic.loc[idx_overlap_plasma].to_csv(folder_data_out / 'ald_metadata_cli.csv')

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
df = pd.read_table(fnames.plasma_aggPeptides, low_memory=False)
N_FRIST_META = 8
df.shape

# %%
df

# %%
df.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = df.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
df = df.set_index(list(df.columns[:N_FRIST_META])).sort_index(axis=1)

# %%
df.loc[:, df.columns.str.contains(VAR_PEP)]

# %%
df.iloc[:20, :6]

# %% [markdown]
# create new multiindex from column

# %%
sep = '.raw.'
df.columns = pd.MultiIndex.from_tuples(df.columns.str.split().str[1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
df = df.stack(0)
df

# %% [markdown]
# ### Index meta data

# %%
meta = df.index.to_frame().reset_index(drop=True)
meta

# %%
meta.describe(include='all')

# %%
id_mappings = ["PEP.StrippedSequence", "PG.ProteinAccessions", "PG.Genes"]
id_mappings = meta[id_mappings].drop_duplicates()
id_mappings.to_csv(folder_data_out / 'ald_plasma_aggPeptides_id_mappings.csv')
id_mappings

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
# After discussing with Lili, `PEP.Quantity` is the fitting entity for
# each unique aggregated Peptide. Duplicated entries are just to drop

# %%
sel_cols = ['Sample ID', 'PEP.StrippedSequence', 'PEP.Quantity']  # selected quantity in last position
df = df.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:2])
df

# %%
df = df.squeeze().dropna().astype(float).unstack()
df

# %%
df = df.dropna(how='all', axis=1)
df

# %%
idx = df.index.to_series()
idx = idx.str.extract(r'(Plate[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %% [markdown]
# - rawfile metadata -> keep

# %%
df = df.set_index(idx)
df_qc = df.loc[idx_qc_plasma].copy()
df = df.loc[idx_overlap_plasma]
df

# %%
# des_data = df.describe() # too slow
des_data = df.isna().sum().to_frame('count').T
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %% [markdown]
# For one raw file no metadata could be extracted (`ERROR: Unable to
# access the RAW file using the native Thermo library.`)

# %%
idx_diff = df.index.difference(raw_meta.index)
annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'peptide number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'peptide measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', feature_name='Aggregated peptides', n_samples=len(df), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_aggPeptides_completness', folder=folder_run)

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
# Dump selected data

# %%
fnames.sel_plasma_aggPeptides = folder_data_out / 'ald_plasma_aggPeptides.pkl'
df.to_pickle(fnames.sel_plasma_aggPeptides)

# %% [markdown]
# Dump QC sample data

# %%
df_qc = df_qc.loc[:, selected]
fnames.qc_plasma_aggPeptides = folder_data_out / 'qc_plasma_aggPeptides.pkl'
df_qc.to_pickle(fnames.qc_plasma_aggPeptides)

# %% [markdown]
# ## Protein Group data

# %%
df = pd.read_csv(fnames.plasma_proteinGroups, low_memory=False)
idx_cols = ['PG.ProteinAccessions', 'PG.Genes']
N_FRIST_META = 3
df

# %%
meta = df.iloc[:, :N_FRIST_META]
meta.describe(include='all')

# %%
id_mappings = ["PG.ProteinAccessions", "PG.Genes"]
id_mappings = meta[id_mappings].drop_duplicates()
id_mappings.to_csv(folder_data_out / 'ald_plasma_proteinGroups_id_mappings.csv', index=False)
id_mappings

# %%
column_types = df.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types  # 'PG.Quantity' expected

# %%
df = df.set_index(list(df.columns[:N_FRIST_META])).sort_index(axis=1)
df.loc[:, df.columns.str.contains(VAR_PG)]


# %% [markdown]
# Drop index columns which are not selected

# %%
def find_idx_to_drop(df: pd.DataFrame, idx_to_keep: list):
    to_drop = [x for x in df.index.names if x not in idx_to_keep]
    logger.info("Columnns to drop: {}".format(",".join((str(x) for x in to_drop))))
    return to_drop


to_drop = find_idx_to_drop(df, idx_cols)
df = df.reset_index(level=to_drop, drop=True)
df.head()

# %% [markdown]
# extract long sample name (highly specific to task)
# - whitespace split, taking last position of column name
# - `sep` splits `Sample ID` from `vars`

# %%
sep = '.raw.'
# sep = '.htrms.'
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
idx = idx.str.extract(r'(Plate[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %%
df = df.set_index(idx)
df_qc = df.loc[idx_qc_plasma].copy()
df = df.loc[idx_overlap_plasma]
df

# %%
df = df.dropna(how='all', axis=0)
df

# %%
des_data = df.describe()
des_data

# %%
freq_feat = des_data.loc["count"].droplevel(-1).rename('freq')
freq_feat.to_csv(folder_data_out / 'freq_ald_plasma_proteinGroups.csv')
# pd.read_csv(folder_data_out / 'freq_ald_plasma_proteinGroups.csv', index_col=0)
freq_feat

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %%
idx_diff = df.index.difference(raw_meta.index)
annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'protein group number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'protein group measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', n_samples=len(df), ax=None, min_feat_prop=.0, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_proteinGroups_completness', folder=folder_run)


# %% [markdown]
# Save unfiltered data
# %%
fname = folder_data_out / 'ald_plasma_proteinGroups_unfiltered.pkl'
df.to_pickle(fname)
fname


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
df.to_pickle(folder_data_out / 'ald_plasma_proteinGroups.pkl')

# %% [markdown]
# Dump QC sample data

# %%
df_qc = df_qc.loc[:, selected].droplevel(1, axis=1)
fnames.qc_plasma_proteinGroups = folder_data_out / 'qc_plasma_proteinGroups.pkl'
df_qc.to_pickle(fnames.qc_plasma_proteinGroups)
df_qc

# %% [markdown]
# # Liver samples

# %% [markdown]
# ## Peptides

# %%
idx_cols = ['PG.ProteinAccessions', 'PG.Genes', 'Sample ID']
N_FRIST_META = 8

# %%
df = pd.read_csv(fnames.liver_aggPeptides, low_memory=False)
df.shape

# %%
df

# %%
df.iloc[:, :N_FRIST_META].describe(include='all')

# %%
column_types = df.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types

# %%
df = df.set_index(list(df.columns[:N_FRIST_META])).sort_index(axis=1)

# %%
df.loc[:, df.columns.str.contains(VAR_PEP)]

# %%
df.columns[:10]

# %% [markdown]
# create new multiindex from column (see examples above)
#
# - split on whitespace
# - select string at first position
# - split on `sep`, keep both

# %%
sep = '.htrms.'
df.columns = pd.MultiIndex.from_tuples(df.columns.str.split().str[1].str.split(
    sep).to_series().apply(tuple), names=['Sample ID', 'vars'])
df = df.stack(0)
df

# %% [markdown]
# ### Index meta data

# %%
meta = df.index.to_frame().reset_index(drop=True)
meta

# %%
id_mappings = ["PEP.StrippedSequence", "PG.ProteinAccessions", "PG.Genes"]
id_mappings = meta[id_mappings].drop_duplicates()
id_mappings.to_csv(folder_data_out / 'ald_liver_aggPeptides_id_mappings.csv')
id_mappings


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
# After discussing with Lili, `PEP.Quantity` is the fitting entity for
# each unique aggregated Peptide. Duplicated entries are just to drop

# %%
sel_cols = ['Sample ID', 'PEP.StrippedSequence', VAR_PEP]  # selected quantity in last position
df = df.reset_index()[sel_cols].drop_duplicates().set_index(sel_cols[:2]).squeeze()
df

# %%
mask = df != 'Filtered'
df = df.loc[mask].astype(float)
df.sort_index()

# %% [markdown]
# Select entry with maximum intensity of `duplicated entries`
#
# > change of variable and many duplicates -> could be PSM table? (close to evidence?)

# %%
mask_idx_duplicated = df.index.duplicated(False)
df.loc[mask_idx_duplicated].sort_index()

# %%
df = vaep.pandas.select_max_by(df=df.reset_index(),
                               grouping_columns=sel_cols[:-1],
                               selection_column=sel_cols[-1]).set_index(sel_cols[:-1])

# %%
assert df.index.duplicated(False).sum() == 0, "Still missing values"

# %%
df = df.unstack()
df

# %%
idx = df.index.to_series()
idx = idx.str.extract(r'(PlateS[\d]_[A-H]\d*)').squeeze()
idx.name = 'Sample ID'
idx.describe()

# %% [markdown]
# - rawfile metadata -> keep

# %%
df = df.set_index(idx)
df_qc = df.loc[idx_qc_liver]
df = df.loc[sel_liver_samples]
df

# %%
df = df.dropna(how='all', axis=1)
df

# %%
# %%time
# des_data = df.describe() unnecessary computation which take too long
des_data = df.isna().sum().to_frame('count').T
des_data

# %% [markdown]
# ### Check for metadata from rawfile overlap

# %% [markdown]
# For one raw file no metadata could be extracted (`ERROR: Unable to
# access the RAW file using the native Thermo library.`)

# %%
# idx_diff = df.index.difference(raw_meta.index)
# annotations.loc[idx_diff]

# %%
kwargs = {'xlabel': 'peptide number ordered by completeness',
          'ylabel': 'peptide was found in # samples',
          'title': 'peptide measurement distribution'}

ax = vaep.plotting.plot_counts(des_data.T.sort_values(by='count', ascending=False).reset_index(
), feat_col_name='count', feature_name='Aggregated peptides', n_samples=len(df), ax=None, **kwargs)

fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='data_liver_aggPeptides_completness', folder=folder_run)

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
# Dump selected data

# %%
fnames.sel_liver_aggPeptides = folder_data_out / 'ald_liver_aggPeptides.pkl'
df.to_pickle(fnames.sel_liver_aggPeptides)

# %%
fnames.qc_liver_aggPeptides = folder_data_out / 'qc_liver_aggPeptides.pkl'
df_qc.to_pickle(fnames.qc_liver_aggPeptides)
df_qc

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
meta = df.iloc[:, :N_FRIST_META]
meta.describe(include='all')

# %%
id_mappings = ["PG.ProteinAccessions", "PG.Genes"]
id_mappings = meta[id_mappings].drop_duplicates()
id_mappings.to_csv(folder_data_out / 'ald_liver_proteinGroups_id_mappings.csv')
id_mappings

# %%
column_types = df.iloc[:, N_FRIST_META:].columns.to_series().apply(lambda s: tuple(s.split('.')[-2:]))
column_types.describe()  # .apply(lambda l: l[-1])

# %%
column_types = ['.'.join(x for x in tup) for tup in list(column_types.unique())]
column_types  # 'PG.Quantity' expected

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
sel_cols = ['PG.ProteinAccessions', 'PG.Genes', 'Sample ID', VAR_PG]  # last one gives quantity
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
df_qc = df.loc[idx_qc_liver]
df = df.loc[sel_liver_samples]
df

# %%
df = df.dropna(how='all', axis=1)
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

# %%
fnames.qc_liver_proteinGroups = folder_data_out / 'qc_liver_proteinGroups.pkl'
df_qc.to_pickle(fnames.qc_liver_proteinGroups)
df_qc

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
