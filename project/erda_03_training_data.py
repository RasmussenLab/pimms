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
# # Build a set of training data
#
# Use a set of (most) common peptides to create inital data sets
#
# - based on `Counter` over all outputs from search (here: MaxQuant)
#    - keep based on threshold `FEAT_COMPLETNESS_CUTOFF` possible features
#    - option: select samples based on `YEARS` (e.g. due constrain by a batch of strains)
#    - collect in wide format data from output files

# %%
from functools import partial
from pathlib import Path
import logging
import multiprocessing

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm_notebook

import vaep

import config


# %%
def join_as_str(seq):
    ret = "_".join(str(x) for x in seq)
    return ret


# %% [markdown]
# ## Setup

# %% [tag=parameters]
RANDOM_SEED: int = 42  # Random seed for reproducibility
FEAT_COMPLETNESS_CUTOFF = 0.25 # Minimal proportion of samples which have to share a feature
SAMPLE_COL = 'Sample ID'
OUT_FOLDER = 'data/selected/'
FN_ID_OLD_NEW: str = 'data/rename/selected_old_new_id_mapping.csv' # selected samples with pride and original id


# %% [markdown]
# Select a specific config file

# %%
# options = ['peptides', 'evidence', 'proteinGroups']
from config.training_data import peptides as cfg
# from config.training_data import evidence as cfg
# from config.training_data import proteinGroups as cfg

cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
cfg_dict


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

# %%
out_folder = Path(OUT_FOLDER) / cfg.NAME
out_folder.mkdir(exist_ok=True, parents=True)


# %% [markdown]
# ## Selected IDs
#
# - currently only `Sample ID` is used
# - path are to `.raw` raw files, not the output folder (could be changed)

# %%
df_ids = pd.read_csv(FN_ID_OLD_NEW)
df_ids

# %% [markdown]
# ## Counter

# %%
counter = CounterClass(FNAME_COUNTER)
counts = counter.get_df_counts()
counts

# %%
if TYPES_COUNT:
    counts = counts.convert_dtypes().astype({'Charge': int}) #
mask = counts['proportion'] >= FEAT_COMPLETNESS_CUTOFF
counts.loc[mask]

# %% [markdown]
# Based on selected samples, retain features that potentially could be in the subset
#
# - if 1000 samples are selected, and given at treshold of 25%, one would need at least 250 observations

# %%
treshold_counts = int(len(df_ids) * FEAT_COMPLETNESS_CUTOFF)
mask = counts['counts'] >= treshold_counts
counts.loc[mask]

# %%
IDX_selected = counts.loc[mask].set_index(cfg.IDX_COLS_LONG[1:]).index
if len(cfg.IDX_COLS_LONG[1:]) > 1:
    IDX_selected = IDX_selected.map(join_as_str)
IDX_selected

# %% [markdown]
# ## Select Dumps

# %%
selected_dumps = df_ids["Sample ID"]
selected_dumps = {k: counter.dumps[k] for k in selected_dumps}
selected_dumps = list(selected_dumps.items())
print(f"Selected # {len(selected_dumps):,d} dumps.")
selected_dumps[:10]


# %% [markdown]
# ## Collect in parallel

# %%
def load_fct(path):
    s = (
    pd.read_csv(path, index_col=cfg.IDX_COLS_LONG[1:], usecols=[*cfg.IDX_COLS_LONG[1:], "Intensity"])
    .squeeze()
    .astype(pd.Int64Dtype())
    )
    if len(cfg.IDX_COLS_LONG[1:]) > 1:
        s.index = s.index.map(join_as_str)
        
    return s
load_fct(selected_dumps[0][-1])


# %%
def collect(folders, index, load_fct):
    current = multiprocessing.current_process()
    i = current._identity[0] % N_WORKERS + 1
    print(" ", end="", flush=True)

    failed = []
    all = pd.DataFrame(index=index)

    with tqdm_notebook(total=len(folders), position=i) as pbar:
        for id, path in folders:
            try:
                s = load_fct(path)
                s.name = id
                all = all.join(s, how='left')
            except FileNotFoundError:
                logging.warning(f"File not found: {path}")
                failed.append((id, path))
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file: {path}")
                failed.append((id, path))
            pbar.update(1)
            
    return all


# %% [markdown]
# ## Collect intensities in parallel

# %%
all = None # free memory

collect_intensities = partial(collect, index=IDX_selected, load_fct=load_fct)

N_WORKERS = 8

with multiprocessing.Pool(N_WORKERS) as p:
    all = list(
        tqdm_notebook(
            p.imap(collect_intensities,
                   np.array_split(selected_dumps, N_WORKERS)),
                   total=N_WORKERS,
        )
    )  
    
all = pd.concat(all, axis=1)
all

# %%
all.memory_usage(deep=True).sum() / (2**20)

# %%
# all = pd.read_pickle('data/selected/proteinGroups/intensities_wide_selected_N00100_M07444.pkl')
all = all.rename(df_ids.set_index("Sample ID")['new_sample_id'], axis=1)
all.head()

# %%
# %%time
fname = out_folder / config.insert_shape(all,  'intensities_wide_selected{}.pkl') 
all.to_pickle(fname)
fname

# %%
# %%time
all.to_csv(fname.with_suffix('.csv'), chunksize=1_000)

# %% [markdown]
# Samples as rows, feature columns as columns
#
# - can fail due to memory -> next notebook
