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
# # Rawfile Renaming
#
# - generated using `workflows/metadata`
# - all raw files collected ~50,000

# %%
from collections import namedtuple
from collections import defaultdict

import yaml
from pathlib import PurePosixPath, Path

import numpy as np
import pandas as pd

import vaep.pandas


def rename(fname, new_sample_id, ext=None):
    fname = PurePosixPath(fname)
    if ext is None:
        ext = fname.suffix
    fname = fname.parent / new_sample_id
    fname = fname.with_suffix(ext)
    return fname



# %% [markdown]
# ## Arguments

# %% tags=["parameters"]
fn_rawfile_metadata: str = 'data/rawfile_metadata.csv' # Machine parsed metadata from rawfile workflow
fn_files_per_instrument: str = 'data/files_per_instrument.yaml' # All parsed raw files nested by instrument (model, attribute, serial number)
fn_files_selected: str = 'data/samples_selected.yaml' # selected files based on threshold of identified peptides
fn_filer_per_instrument_selected: str = 'data/files_selected_per_instrument.yaml' # Selected parsed raw files nested by instrument (model, attribute, serial number)
out_folder: str = 'data/rename'

# %%
out_folder = Path(out_folder)
out_folder.mkdir(exist_ok=True)

files_out = dict()

# %% [markdown]
# ### Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %%
df_meta = pd.read_csv(fn_rawfile_metadata, header=[0, 1], index_col=0)
date_col = ('FileProperties', 'Content Creation Date')
df_meta[date_col] = pd.to_datetime(
    df_meta[date_col])
df_meta.sort_values(date_col, inplace=True)
df_meta
msg = f"A total of {len(df_meta)} raw files could be read using the ThermoFisherRawFileParser." 

# %%
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats.T

# %% [markdown]
# # Erda Paths

# %%
cols_identifies = [('FileProperties', 'Pathname'),
 ('FileProperties', 'Version'),
 ('FileProperties', 'Content Creation Date'),
 ('InstrumentProperties', 'Thermo Scientific instrument model'),
 ('InstrumentProperties', 'instrument attribute'),
 ('InstrumentProperties', 'instrument serial number'),
 ('InstrumentProperties', 'Software Version'),
 ('InstrumentProperties', 'firmware version'),
]

df_meta = df_meta[cols_identifies]
df_meta.columns = [t[-1] for t in cols_identifies]
df_meta

# %% [markdown]
# Replace `tmp/` with `./` (artefact)

# %%
df_meta['Pathname'] = df_meta['Pathname'].str.replace('tmp/', './')

# %% [markdown]
# Create new sample identifier

# %%
idx_all = (pd.to_datetime(df_meta["Content Creation Date"]).dt.strftime("%Y_%m_%d_%H_%M")
        + '_'
        + df_meta["Thermo Scientific instrument model"].str.replace(' ', '-')
        + '_'
        + df_meta["instrument serial number"].str.split('#').str[-1]).str.replace(' ', '-')

mask = idx_all.duplicated(keep=False)
duplicated_sample_idx = idx_all.loc[mask].sort_values()  # duplicated dumps
duplicated_sample_idx

# %%
df_meta['new_sample_id'] =  idx_all


df_meta["Path_new"] = df_meta[["Pathname", "new_sample_id"]].apply(lambda s: rename(*s), axis=1)


_n = df_meta.groupby("new_sample_id").cumcount().astype('string').str.replace('0', '')
_n[_n != ''] = '_r' + _n[_n != '']
_n.value_counts()

df_meta.loc[mask, "new_sample_id"] = df_meta.loc[mask, "new_sample_id"] + _n


df_meta.loc[mask, ["Pathname", "new_sample_id"]]

# %%
df_meta.loc[~mask, ["Pathname", "new_sample_id"]]

# %%
df_meta["Path_new"] = df_meta[["Pathname", "new_sample_id"]].apply(lambda s: rename(*s), axis=1)

# %%
assert df_meta["Pathname"].is_unique
assert df_meta["Path_new"].is_unique
assert df_meta["new_sample_id"].is_unique

# %% [markdown]
# ### Save new paths to disk

# %%
df_meta["Path_old"] = df_meta["Pathname"]

df_meta[["Path_old", "Path_new", "new_sample_id"]]

# %% [markdown]
# ## Selected Files

# %%
with open(fn_files_selected) as f:
    files_selected = yaml.safe_load(f)
print(f'Threshold: {files_selected["threshold"]:,d}')


# %%
df_meta.loc[files_selected["files"]]

# %%
mask = idx_all.duplicated()
selected = df_meta.loc[~mask].index.intersection(files_selected["files"])
df_meta.loc[selected]

# %%
fname = out_folder / 'selected_old_new_id_mapping.csv'
files_out[fname.name] = fname
df_meta.loc[selected, ["Path_old", "Path_new", "new_sample_id"]].to_csv(fname)
fname

# %% [markdown]
# ### OS rename

# %%
# df_meta["Path_old"] = df_meta["Pathname"]
df_meta.loc[selected][["Path_old", "Path_new", "new_sample_id"]]

# %%
import os

# %%
# For all file names?
# os.rename?

# %%
# For MQ output?
# os.renames?

# %% [markdown]
# Selected path_old -> path_new

# %% [markdown]
# ### SFTP command

# %%
commands = df_meta.loc[selected]
commands = '!put ' + commands['Path_old'].astype('string') + ' ' + commands['Path_new'].astype('string')
print(commands.sample(10).to_csv(sep=' ', header=False, index=False))

# %% [markdown]
# write all

# %%
fname = out_folder / 'sftp_commands_rawfiles'
commands.to_csv(fname, sep=' ', header=False, index=False)

# %% [markdown]
# ## Put output files on PRIDE
#
# - `mq_out` folder
# - move from `Sample ID` folder into `new_sample_id` on erda

# %%
commands = df_meta.loc[selected]
commands = '!put ' + commands.index + '/* ' + commands["new_sample_id"] + '/*'
  
print(commands.sample(10).to_csv(sep=' ', header=False, index=False))

# %%
fname = out_folder / 'sftp_commands_mq_output'
commands.to_csv(fname, sep=' ', header=False, index=False)
