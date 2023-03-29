# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rawfile Renaming
#
# - generated using `workflows/metadata`
# - all raw files collected ~50,000

# %%
from collections import defaultdict, namedtuple
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import vaep.pandas
import yaml


def rename(fname, new_sample_id, new_folder=None, ext=None):
    fname = PurePosixPath(fname)
    if ext is None:
        ext = fname.suffix
    if new_folder is None:
        new_folder = fname.parent
    else:
        new_folder = PurePosixPath(new_folder)
    fname = new_folder / new_sample_id
    fname = fname.with_suffix(ext)
    return fname.as_posix()


# %% [markdown]
# ## Arguments

# %% tags=["parameters"]
fn_rawfile_metadata: str = 'data/rawfile_metadata.csv' # Machine parsed metadata from rawfile workflow
fn_files_selected: str = 'data/samples_selected.yaml' # selected files based on threshold of identified peptides
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
df_meta = pd.read_csv(fn_rawfile_metadata, header=[0, 1], index_col=0, low_memory=False)
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

# %%
df_meta["Instrument_name"] = (
    df_meta["Thermo Scientific instrument model"].str.replace(' ', '-')
    + '_'
    + df_meta["instrument serial number"].str.split('#').str[-1]
).str.replace(' ', '-')

df_meta["Instrument_name"].value_counts().index

# %% [markdown]
# Create new sample identifier

# %%
date_col = "Content Creation Date"
idx_all = (pd.to_datetime(df_meta[date_col]).dt.strftime("%Y_%m_%d_%H_%M")
        + '_'
        + df_meta["Instrument_name"]
).str.replace(' ', '-')

mask = idx_all.duplicated(keep=False)
duplicated_sample_idx = idx_all.loc[mask].sort_values()  # duplicated dumps
duplicated_sample_idx

# %%
df_meta['new_sample_id'] =  idx_all


df_meta["Path_new"] = df_meta[["Pathname", "new_sample_id", "Instrument_name"]].apply(lambda s: rename(*s), axis=1)


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

# %%
df_meta

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
def build_instrument_name(s):
    """Process in order, only keep one name"""
    ret = ''
    used_before = set()
    for string_w_withspaces in s:
        strings_ = string_w_withspaces.split()
        for string_ in strings_:
            if string_ not in used_before:
                ret += f'_{string_}'
        used_before |= set(strings_)
    ret = (ret[1:] # remove _ from start
           .replace('Slot_#', '')
           .replace('slot_#', '')
          )
    return ret


(df_meta[
        [
            "Thermo Scientific instrument model",
            "instrument attribute",
            "instrument serial number",
        ]
    ]
    .sample(20)
    .apply(build_instrument_name, axis=1)
)

# %%
fname = out_folder / 'selected_old_new_id_mapping.csv'
files_out[fname.name] = fname.as_posix()
df_meta.loc[selected, ["Path_old", "Path_new", "new_sample_id"]].to_csv(fname)
fname

# %% [markdown]
# ### OS rename

# %%
# df_meta["Path_old"] = df_meta["Pathname"]
df_meta.loc[selected][["Path_old", "Path_new", "new_sample_id"]]

# %% [markdown]
# ## Put files on PRIDE FTP server
#
# rename using `new_sample_id`

# %% [markdown]
# ### LFTP commands - raw files
#
# `-f` option allows to pass commands from a file
# One needs to at least an `open` as the first line to log in to an ftp server
# For pride one needs to additionally `cd` to the correct folder:
# ```bash
# > open ...
# > cd ...
# ```
# to allow parallell commands, use the runtime setting
# ```bash
# >>> cat ~/.lftprc 
# set cmd:parallel 2
# ```

# %% [markdown]
# Create folders on pride for raw files

# %%
df_meta["folder_raw"] = "./raw_files/" + df_meta["Instrument_name"]
df_meta["folder_raw"].unique()

fname = out_folder / 'raw_file_directories.txt'

commands = 'mkdir -p ' + df_meta.loc[selected, "folder_raw"].drop_duplicates()
commands.to_csv(fname, header=False, index=False)

# %% [markdown]
# Create upload commands of raw files to create folders (could be combined with above)

# %%
commands = df_meta.loc[selected]
commands = (
    'put ' 
    + commands['Path_old'].astype('string')
    + ' -o ' 
    + "./raw_files/" 
    + commands["Instrument_name"] 
    + '/'
    + commands['new_sample_id'] + '.raw'
)
print(commands.sample(10).to_csv(sep=' ', header=False, index=False))


# %% [markdown]
# write all to file

# %%
fname = out_folder / 'lftp_commands_rawfiles.txt'
commands.to_csv(fname, header=False, index=False)

# %% [markdown]
# ### LFTP commands - MaxQuant output

# %% [markdown]
# Create upload commands of MaxQuant output folders to pride using mirror
#
# - `mq_out` folder
# - move from `Sample ID` folder into `new_sample_id` on erda

# %%
commands = df_meta.loc[selected]
commands = (
    "mirror -R --only-missing -P 8 --exclude-glob *.pdf " # command
    + "mq_out/" + commands.index # source
    # + "./" + pd.to_datetime(commands[date_col]).dt.strftime("%Y") + "/" + commands["new_sample_id"] # dest
    + " ./MQ_tables/" + commands["Instrument_name"]+ "/" + commands["new_sample_id"] # dest
)

print(commands.sample(10).to_csv(header=False, index=False))

# %% [markdown]
# write all to file

# %%
fname = out_folder / 'lftp_commands_mq_output.txt'
commands.to_csv(fname, header=False, index=False)

# %%
