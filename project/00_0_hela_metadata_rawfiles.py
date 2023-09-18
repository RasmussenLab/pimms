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
# # Rawfile metadata
#
# - generated using `workflows/metadata`
# - all raw files collected ~50,000

# %%
from collections import namedtuple
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd

import vaep.pandas

# %% [markdown]
# ## Arguments

# %% tags=["parameters"]
fn_rawfile_metadata: str = 'data/rawfile_metadata.csv'  # Machine parsed metadata from rawfile workflow
# outputs
# All parsed raw files nested by instrument (model, attribute, serial number)
fn_files_per_instrument: str = 'data/files_per_instrument.yaml'
fn_files_selected: str = 'data/samples_selected.yaml'  # selected files based on threshold of identified peptides
# Selected parsed raw files nested by instrument (model, attribute, serial number)
fn_files_per_instrument_selected: str = 'data/files_selected_per_instrument.yaml'

# %% [markdown]
# ### Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %%
df_meta_rawfiles = pd.read_csv(fn_rawfile_metadata, header=[0, 1], index_col=0, low_memory=False)
date_col = ('FileProperties', 'Content Creation Date')
df_meta_rawfiles[date_col] = pd.to_datetime(
    df_meta_rawfiles[date_col])
df_meta_rawfiles.sort_values(date_col, inplace=True)
df_meta_rawfiles

# %%
msg = f"A total of {len(df_meta_rawfiles)} raw files could be read using the ThermoFisherRawFileParser."
print(msg)

# %%
meta_stats = df_meta_rawfiles.describe(include='all', datetime_is_numeric=True)
meta_stats.T

# %% [markdown]
# subset with variation

# %%
meta_stats.loc[:, (meta_stats.loc['unique'] > 1) | (meta_stats.loc['std'] > 0.1)].T

# %%
# needs to go to Config which is not overwriteable by attribute selection
df_meta_rawfiles_columns = df_meta_rawfiles.columns
meta_raw_names = df_meta_rawfiles.columns.droplevel()
assert meta_raw_names.is_unique
df_meta_rawfiles.columns = meta_raw_names
df_meta_rawfiles

# %%
meta_raw_selected = [
    'Content Creation Date',
    'Thermo Scientific instrument model',
    'instrument serial number',
    'Software Version',
    'Number of MS1 spectra',
    'Number of MS2 spectra',
    'Number of scans',
    'MS max charge',
    'MS max RT',
    'MS min MZ',
    'MS max MZ',
    'MS scan range',
    'mass resolution',
    'Retention time range',
    'Mz range',
    'beam-type collision-induced dissociation',
    'injection volume setting',
    'dilution factor',
]
df_meta_rawfiles[meta_raw_selected].describe(percentiles=np.linspace(0.05, 0.95, 10))

# %% [markdown]
# - `MS min MZ`: outlier clearly shifts means
# - `mass resolution` is unique (can this be?)
# - `dillution factor` is unique (can this be?)

# %% [markdown]
# ## Instrument type and settings
#
# check some columns describing settings
#   - quite some variation due to `MS max charge`: Is it a parameter?

# %%
MetaRawSettings = namedtuple(
    'MetaRawSettings',
    'ms_model ms_attr ms_sn ms_firmware max_charge mass_res cid_type inject_volume dill_factor')
meta_raw_settings = [
    'Thermo Scientific instrument model',
    'instrument attribute',
    'instrument serial number',
    'Software Version',
    'MS max charge',
    'mass resolution',
    'beam-type collision-induced dissociation',
    'injection volume setting',
    'dilution factor',
]
meta_raw_settings = MetaRawSettings(*meta_raw_settings)
meta_raw_settings

# %%
# index gives first example with this combination
# df_meta_rawfiles[list(meta_raw_settings)].drop_duplicates()
df_meta_rawfiles[list(meta_raw_settings)].drop_duplicates(ignore_index=True)

# %% [markdown]
# view without `MS max charge`:
#   - software can be updated
#   - variation by `injection volume setting` and instrument over time
#   - missing `dilution factor`
#

# %%
to_drop = ['MS max charge']
# df_meta_rawfiles[list(meta_raw_settings)].drop(to_drop,
# axis=1).drop_duplicates(ignore_index=False) # index gives first example
# with this combination
df_meta_rawfiles[list(meta_raw_settings)].drop(to_drop, axis=1).drop_duplicates(ignore_index=True)

# %% [markdown]
# Relatively big samples for different machines of the same kind running with the same firmware:

# %%
df_meta_rawfiles.groupby([meta_raw_settings.ms_model, meta_raw_settings.ms_firmware])[
    meta_raw_settings.ms_model].count().sort_values().tail(10)

# %% [markdown]
# Ignoring instrument software

# %%
grouping = df_meta_rawfiles.groupby(list(meta_raw_settings[:3]))
instrument_counts = grouping[meta_raw_settings.ms_model].count().sort_values()
msg += (f" There are a total of {len(instrument_counts)} unique instruments in the entire dataset (based on the instrument name, attributs and serial number)"
        f", of which at least {(instrument_counts >= 1000).sum()} have 1,000 raw files assigned to them. Note that the entire dataset contains fractionated measurements.")
instrument_counts

# %%
ms_groups = vaep.pandas.create_dict_of_dicts(grouping.groups, verbose=True, transform_values=list)

# %%
# d = dict()
# for (k1, k2, k3), v in grouping.groups.items():
#     print(f"{str((k1,k2,k3)):90}: {len(v):>5}")
#     if not k1 in d:
#         d[k1] = dict()
#     if not k2 in d[k1]:
#         d[k1][k2] = dict()
#     d[k1][k2][k3] = list(v)
# assert ms_groups == d

# %% [markdown]
# Save selection yaml

# %%
with open(fn_files_per_instrument, 'w') as f:
    yaml.dump(ms_groups, f)

# %% [markdown]
# ## Quantified files
#
# - export nested files with quantified files based on selection based on identified peptides threshold

# %%
with open(fn_files_selected) as f:
    files_selected = yaml.safe_load(f)
print(f'Threshold: {files_selected["threshold"]:,d}')

# %% [markdown]
# - save metadata for selected, quantified samples / raw files

# %%
df_meta_rawfiles.loc[files_selected['files']].to_csv('data/files_selected_metadata.csv')

# %%
grouping = df_meta_rawfiles.loc[files_selected['files']].groupby(list(meta_raw_settings[:3]))
instrument_counts = grouping[meta_raw_settings.ms_model].count().sort_values()
N = 500
msg += (
    f" Among the {len(files_selected['files'])} raw files with a minimum of {files_selected['threshold']:,d} identified peptides there are a total of {len(instrument_counts)} unique instruments with quantified runs"
    f", of which {(instrument_counts >= N).sum()} have at least {N:,d} rawfiles assigned to them.")
instrument_counts.to_csv('data/files_selected_per_instrument_counts.csv')
instrument_counts.to_frame('No. samples')

# %%
ms_groups = vaep.pandas.create_dict_of_dicts(grouping.groups, verbose=True, transform_values=list)
with open(fn_files_per_instrument_selected, 'w') as f:
    yaml.dump(ms_groups, f)

# %%
print(msg)
