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

# %% [markdown]
# # Selected files
#
# - document metadata and file sizes of published dataset in Scientific Data Report 
#
# ## Contents
#
# 1. Number of files per instrument
# 2. Rawfile sizes per instrument
# 3. peptide - rawfile map (protein group, precursor)?
#     - based on selected samples

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn


from vaep.io import thermo_raw_files
import vaep.pandas

# %% [markdown]
# ## PARAMETERS

# %%
fn_id_old_new: str = 'data/rename/selected_old_new_id_mapping.csv' # selected samples with pride and original id
fn_raw_file_size: str = 'processed/all_raw_file_sizes.csv' # raw file sizes
fn_rawfile_metadata: str = 'data/rawfile_metadata.csv'
fn_summaries:str = 'data/processed/all_summaries.json'
date_col:str = 'Content Creation Date'
out_folder: str = 'data/dev_datasets/pride_upload'

# %% [markdown]
# ## Prepare outputs

# %%
out_folder = Path(out_folder)
out_folder.mkdir(exist_ok=True)
files_out = dict()

# %% [markdown]
# ## ID mapping
#
# - artefact of local vs pride data

# %%
df_ids = pd.read_csv(fn_id_old_new, index_col=0)
df_ids

# %%
df_ids.index.is_unique

# %% [markdown]
# ## Raw file sizes

# %%
df_raw_file_size = pd.read_csv(fn_raw_file_size, index_col=0)
df_raw_file_size

# %%
df_raw_file_size.index.is_unique

# %%
from pathlib import Path
df_raw_file_size['path'] = df_raw_file_size['path'].apply(lambda x: Path(x).as_posix())
df_raw_file_size = df_raw_file_size.reset_index().set_index('path')
df_raw_file_size

# %%
df_raw_file_size = df_raw_file_size.loc[df_ids['Path_old'].str[2:].to_list()]
df_raw_file_size

# %%
df_raw_file_size = df_raw_file_size.reset_index().set_index('name')

# %% [markdown]
# ## Raw file metadata extracted from ThermoRawFileParser

# %%
df_meta = pd.read_csv(fn_rawfile_metadata, header=[0, 1], index_col=0, low_memory=False)
assert df_meta.index.is_unique
df_meta

# %%
df_meta = df_meta.loc[df_ids.index]
df_meta.columns = df_meta.columns.droplevel() # remove top level name
df_meta

# %% [markdown]
# ## Summary files from MaxQuant search

# %%
df_summaries = pd.read_json(fn_summaries, orient='index')
assert df_summaries.index.is_unique
df_summaries = df_summaries.loc[df_meta.index]
df_summaries

# %% [markdown]
# # Combine data and dump

# %%
df_meta = (df_ids
           .join(df_raw_file_size)
           .join(df_meta)
           .join(df_summaries)
          )
df_meta

# %%
df_meta = df_meta.set_index('new_sample_id')
df_meta.index.name = 'Sample ID'

# %%
df_meta = (df_meta
           .drop(['Path_old', 'Pathname', 'path'], axis=1)
           .rename({'Path_new':'Pathname'}, axis=1)
           .dropna(how='all', axis=1)
           .convert_dtypes()
           .assign(**{date_col: lambda df_meta: pd.to_datetime(df_meta[date_col])})
)
df_meta

# %% [markdown]
# Save curated data for dumped files

# %%
fname = out_folder / 'pride_metadata.csv'
files_out[fname.name] = fname.as_posix()
df_meta.to_csv(fname)

fname = out_folder / 'pride_metadata_schema.json'
files_out[fname.name] = fname.as_posix()
df_meta.dtypes.astype('string').to_json(fname)

# %% [markdown]
# # Analysis

# %% [markdown]
# How to load dumped file

# %%
dtypes = pd.read_json(
    files_out['pride_metadata_schema.json'],
    orient='index'
    ).squeeze()
mask_dates = dtypes.str.contains('datetime') # date columns need to be provide separately
pd.read_csv(files_out['pride_metadata.csv'],
            parse_dates=mask_dates.loc[mask_dates].index.to_list(),
            dtype=dtypes.loc[~mask_dates].to_dict()
).dtypes

# %% [markdown]
# ## Output Excel for Analysis

# %%
writer_args = dict(float_format='%.3f')
fname = out_folder / 'pride_data_infos.xlsx'
files_out[fname.name] = fname.as_posix()
excel_writer = pd.ExcelWriter(fname)

# %% [markdown]
# ## Varying data between runs

# %%
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats.T.to_excel(excel_writer, sheet_name='des_stats', **writer_args)

view = meta_stats.loc[:, (meta_stats.loc['unique'] > 1) |  (meta_stats.loc['std'] > 0.01)].T
view.to_excel(excel_writer, sheet_name='des_stats_varying', **writer_args)

# %% [markdown]
# ## Instruments in selection

# %%
thermo_raw_files.cols_instrument

# %%
df_meta[date_col] = pd.to_datetime(df_meta[date_col])

counts_instrument = (df_meta
                     .groupby(thermo_raw_files.cols_instrument)[date_col]
                     .agg(['count', 'min', 'max'])
                     .sort_values(by=thermo_raw_files.cols_instrument[:2] + ['count'], ascending=False))
counts_instrument.to_excel(excel_writer, sheet_name='instruments', **writer_args)
counts_instrument

# %%
top10_instruments = counts_instrument['count'].nlargest(10)
top10_instruments

# %%
mask_top10_instruments = df_meta[thermo_raw_files.cols_instrument].apply(lambda x: tuple(x) in top10_instruments.index, axis=1)
assert mask_top10_instruments.sum() == top10_instruments.sum()

# %% [markdown]
# ## File size and number of identifications

# %%
cols = ['Peptide Sequences Identified', 'size_gb']

mask = ((df_meta[cols[0]] < 20_000) & (df_meta[cols[1]] > 3.5)
        | (df_meta[cols[1]] > 5)
       )

# df_meta[thermo_raw_files.cols_instrument + cols].query(f'({cols[1]} > 3.5'
#                     f' & `{cols[0]}` < 25_000)'
#                     f' | {cols[1]} > 5',
#                     engine='python')


cols = ['Peptide Sequences Identified', 'size_gb']
ax = df_meta.loc[~mask, cols].plot.scatter(cols[0], cols[1], label='not selected')
ax = df_meta.loc[mask, cols].plot.scatter(cols[0],  cols[1], color='orange', label='selected', ax=ax)

view = df_meta.loc[mask, thermo_raw_files.cols_instrument + cols].sort_values(by=cols)
view.to_excel(excel_writer, sheet_name='instrument_outliers', **writer_args)
view

# %%
cols = ['Number of MS1 spectra', 'Number of MS2 spectra',
        'Peptide Sequences Identified']
cols = vaep.pandas.get_columns_accessor_from_iterable(cols)

view = df_meta.loc[mask_top10_instruments]
# view = df_meta


fig, ax = plt.subplots(figsize=(18,12))

fig = ax.get_figure()

ax = seaborn.scatterplot(view,
                    x=cols.Number_of_MS1_spectra,
                    y=cols.Number_of_MS2_spectra,
                    hue='instrument serial number',
                    legend='brief',
                    ax=ax,
                    palette='deep')
l = ax.legend(loc='right', bbox_to_anchor=(1.35, 0.5), ncol=1)

fname = out_folder / 'ms1_to_ms2_top10_instruments.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(fig, fname)
# fig.savefig(fname)
# fig.savefig(fname.with_suffix('.png'), dpi=600)

# %%
ax = view.plot.scatter(x=cols.Peptide_Sequences_Identified,
                          y=cols.Number_of_MS1_spectra,
                          label=cols.Number_of_MS1_spectra,
                          c='green')
ax = view.plot.scatter(x=cols.Peptide_Sequences_Identified,
                          y=cols.Number_of_MS2_spectra,
                          label=cols.Number_of_MS2_spectra,
                          ylabel='# spectra',
                          ax=ax)

# %% [markdown]
# ## run length to number of identified peptides

# %%
df_meta.filter(like='RT', axis=1).describe()

# %%
cols = ['MS max RT',
        'Peptide Sequences Identified']
cols = vaep.pandas.get_columns_accessor_from_iterable(cols)

ax = view.plot.scatter(cols.MS_max_RT, cols.Peptide_Sequences_Identified)

ax = ax = seaborn.scatterplot(
                    view,
                    x=cols.MS_max_RT,
                    y=cols.Peptide_Sequences_Identified,
                    hue='instrument serial number',
                    legend='brief',
                    ax=ax,
                    palette='deep')
l = ax.legend(loc='right', bbox_to_anchor=(1.4, 0.5), ncol=1)

fname = out_folder / 'RT_vs_identified_peptides_top10_instruments.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ## Outputs

# %%
excel_writer.close()

# %%
files_out

# %%
