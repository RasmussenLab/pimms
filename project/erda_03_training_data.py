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
# # Build a set of training data
#
# Use a set of (most) common peptides to create inital data sets

# %%
import yaml
import json
import random  # shuffle, seed
import functools
from pathlib import Path
import logging

import pandas as pd
from tqdm.notebook import tqdm

from vaep.io import data_objects

import config as config

# %%
from typing import List
def select_files_by_parent_folder(fpaths:List, years:List):
    selected = []
    for year_folder in years:
        # several passes, but not a bottle neck
        selected += [dump for dump in fpaths if year_folder in dump.parent.stem]
    return selected


# %% [markdown]
# ## Setup

# %%
RANDOM_SEED: int = 42  # Random seed for reproducibility

FEAT_COMPLETNESS_CUTOFF = 0.25 # Minimal proportion of samples which have to share a feature

YEARS = ['2017','2018', '2019', '2020']

SAMPLE_COL = 'Sample ID'

# %% [markdown]
# Select a specific config file

# %%
# options = ['peptides', 'evidence', 'proteinGroups']
from config.training_data import peptides as cfg
# from config.training_data import evidence as cfg
# from config.training_data import proteinGroups as cfg

{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}

# %% [markdown]
# Set defaults from file (allows to potentially overwrite parameters)

# %%
# normal structure of config.py files
NAME = cfg.NAME
BASE_NAME = cfg.BASE_NAME

TYPES_DUMP = cfg.TYPES_DUMP
TYPES_COUNT = cfg.TYPES_COUNT

IDX_COLS_LONG = cfg.IDX_COLS_LONG

LOAD_DUMP = cfg.LOAD_DUMP

CounterClass = cfg.CounterClass
FNAME_COUNTER = cfg.FNAME_COUNTER

# %% [markdown]
# ## Counter

# %%
counter = CounterClass(FNAME_COUNTER)
counts = counter.get_df_counts()

if TYPES_COUNT:
    counts = counts.convert_dtypes().astype({'Charge': int}) #
mask = counts['proportion'] >= FEAT_COMPLETNESS_CUTOFF
counts.loc[mask]

# %% [markdown]
# ## Selected Features
#
# - index names should also match!
# - if not-> rather use a list?

# %%
selected_features = counts.loc[mask].set_index(counter.idx_names).sort_index().index
# selected_features.name = 'Gene names' # needs to be fixed
selected_features

# %% [markdown]
# ## Select Dumps

# %%
selected_dumps = select_files_by_parent_folder(list(counter.dumps.values()), years=YEARS)
print("Total number of files:", len(selected_dumps))
selected_dumps[-10:]

# %% [markdown]
# ## Load one dump
#
# - check that this looks like you expect it
#

# %%
LOAD_DUMP(selected_dumps[0])

# %% [markdown]
# ## Process folders
#
# - potentially in parallel, aggregating results
# - if needed: debug using two samples

# %%
from typing import List, Callable
from pandas.errors import EmptyDataError
def process_folders(fpaths: List[Path],
                    selected_features: pd.Index,
                    load_folder: Callable,
                    id_col='Sample ID',
                    dtypes: dict = {
                        'Sample ID': 'category',
                        'Sequence': 'category'}) -> tuple:
    print(f"started new process with {len(fpaths)} files.")
    data_intensity = []
    for i, fpath in enumerate(fpaths):
        if not i % 10: print(f"File ({i}): {fpath}")
        sample_name = fpath.stem
        try:
            dump = load_folder(fpath)
        except EmptyDataError:
            logging.warning(f'Empty dump: {fpath}')
            continue
        except FileNotFoundError:
            logging.warning(f'Missing dump: {fpath}')
            continue
        sequences_available = dump.index.intersection(selected_features)
        dump = dump.loc[sequences_available, 'Intensity'].reset_index()
        dump[id_col] = sample_name
        dump = dump.astype(dtypes)
        data_intensity.append(dump)
    
    data_intensity = pd.concat(data_intensity, copy=False, ignore_index=True)
    data_intensity = data_intensity.astype(dtypes)
    return data_intensity

# # experiment
# process_folders(selected_dumps[:2],
#                 selected_features=selected_features,
#                 load_folder=LOAD_DUMP,
#                 dtypes=TYPES_DUMP)



# %%
# %%time

process_folders_peptides = functools.partial(process_folders,
                                             selected_features=selected_features,
                                             load_folder=LOAD_DUMP,
                                             dtypes=TYPES_DUMP)
collected_dfs = data_objects.collect_in_chuncks(paths=selected_dumps,
                                                process_chunk_fct=process_folders_peptides,
                                                chunks=200,
                                                n_workers=1 # to debug, don't multiprocess
                                               )

# one would need to aggregate categories first to keep them during aggregation?
collected_dfs = pd.concat(collected_dfs, copy=False, ignore_index=True)
collected_dfs = collected_dfs.astype(TYPES_DUMP)
df_intensities = collected_dfs
df_intensities

# %% [markdown]
# Except Intensities everything should be of data type category in order to save memory

# %%
df_intensities.dtypes

# %%
df_intensities.describe(include='all')

# %% [markdown]
# Check how many samples could be loaded, set total number of features

# %%
N = df_intensities[SAMPLE_COL].nunique()
M = len(selected_features)
N,M

# %% [markdown]
# set index columns provided and squeeze to series

# %%
df_intensities = df_intensities.set_index(IDX_COLS_LONG).squeeze()

# %%
base_name = f'{BASE_NAME}_' + '_'.join(YEARS)
fname = config.FOLDER_DATA / config.insert_shape(df_intensities,  base_name + '{}.pkl', shape=(N,M))
print(f"{fname = }")
df_intensities.to_pickle(fname)

# %% [markdown]
# - Only binary pickle format works for now
# - csv and reshaping the data needs to much memory for a single erda instance with many samples

# %%
# df_intensities = df_intensities.unstack(['Sample ID'])
# df_intensities

# %%
# df_intensities.sort_index(inplace=True)
# base_name = "df_intensities_evidence_long" + '_'.join(YEARS)
# fname = config.FOLDER_DATA / config.insert_shape(df_intensities, base_name + '{}.csv', shape=(N,M))
# print(f"{fname = }")
# df_intensities.to_csv(fname)
# df_intensities
