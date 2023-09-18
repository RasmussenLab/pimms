# ---
# jupyter:
#   jupytext:
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
import pandas as pd


# %% [markdown]
# ## PARAMETERS

# %%
fn_id_old_new: str = 'data/rename/selected_old_new_id_mapping.csv'  # selected samples with pride and original id
fn_raw_file_size: str = 'processed/all_raw_file_sizes.csv'  # raw file sizes
fn_rawfile_metadata: str = 'data/rawfile_metadata.csv'
fn_summaries: str = 'data/processed/all_summaries.json'
date_col: str = 'Content Creation Date'
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
df_meta.columns = df_meta.columns.droplevel()  # remove top level name
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
           .rename({'Path_new': 'Pathname'}, axis=1)
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
mask_dates = dtypes.str.contains('datetime')  # date columns need to be provide separately
pd.read_csv(files_out['pride_metadata.csv'],
            parse_dates=mask_dates.loc[mask_dates].index.to_list(),
            dtype=dtypes.loc[~mask_dates].to_dict()
            ).dtypes


# %%
files_out

# %%
