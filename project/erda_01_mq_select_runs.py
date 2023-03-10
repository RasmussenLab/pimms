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

# %% [markdown] Collapsed="false"
# # MaxQuant (MQ) Output-Files
#
# Files compared:
# 1. `Summary.txt`
# 2. `mqpar.xml`
# 3. `peptides.txt`
# 4. `proteins.txt`
#
# There is are many files more, where several files seem to be available in several times in different formats.

# %%
import sys
import logging
from pathlib import Path, PurePosixPath
import yaml
import random

##################
### Logging ######
##################

# Setup logging in notebooks
from vaep.logging import setup_nb_logger
setup_nb_logger()
logger = logging.getLogger()

logging.info('Start with handlers: \n' + "\n".join(f"- {repr(log_)}" for log_ in logger.handlers))

### Other imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets

from vaep.io.mq import MaxQuantOutputDynamic
from vaep import plotting

from vaep.io import data_objects
from vaep.io.data_objects import MqAllSummaries 

##################
##### CONFIG #####
##################
import config
from config import FOLDER_MQ_TXT_DATA, FOLDER_PROCESSED

ELIGABLE_FILES_YAML = Path('config/eligable_files.yaml')
MAP_FOLDER_PATH = Path('config/file_paths')
FPATH_ALL_SUMMARIES = FOLDER_PROCESSED / 'all_summaries.json'
FN_RAWFILE_METADATA = 'data/rawfile_metadata.csv'

from config import FOLDER_DATA # project folder for storing the data
logger.info(f"Search Raw-Files on path: {FOLDER_MQ_TXT_DATA}")

# %% Collapsed="false"
folders = [folder for folder in  Path(FOLDER_MQ_TXT_DATA).iterdir() if folder.is_dir() and not folder.name.startswith('.')]

# %% Collapsed="false"
folders_dict = {folder.name: folder for folder in sorted(folders)}
assert len(folders_dict) == len(folders), "Non unique file names"

with open(MAP_FOLDER_PATH, 'w') as f:
    yaml.dump({ k: str(PurePosixPath(v)) for k, v in folders_dict.items()} , f)
logger.info(f"Save map of file names to file paths to: {str(MAP_FOLDER_PATH)}")

# w_file = widgets.Dropdown(options=[folder for folder in folders], description='View files')
w_file = widgets.Dropdown(options=folders_dict, description='View files')
w_file

# %%
mq_output = MaxQuantOutputDynamic(w_file.value)
mq_output

# %%
print(f"Results will be saved in subfolders in\n\t{str(FOLDER_PROCESSED.absolute())}"
      "\nusing the name of the specified input-folder per default. Change to your liking.")

# %% [markdown]
# > Go to the block you are interested in!

# %% [markdown] Collapsed="false"
# ### Summaries Data

# %%
# %%time
pd.options.display.max_columns = 49
mq_all_summaries = MqAllSummaries(FPATH_ALL_SUMMARIES)
mq_all_summaries.load_new_samples(folders=folders)

# %% Collapsed="false"
if mq_all_summaries.empty_folders:
    print(mq_all_summaries.empty_folders)
    with open('log_empty_folder.txt', 'a') as f:
        f.writelines(mq_all_summaries.empty_folders)
print(f"In total processed: {len(mq_all_summaries):5}")

# %% Collapsed="false"
pd.options.display.max_columns = len(mq_all_summaries.df.columns)

# %%
mq_all_summaries.df.info()


# %% [markdown]
# - SIL - MS2 based on precursor which was a set of peaks
# - PEAK - MS2 scan based on a single peak on precursor spectrum
# - ISO - isotopic pattern detection
#

# %%
class col_summary:
    MS1 = 'MS'
    MS2 = 'MS/MS' 
    MS2_identified  = 'MS/MS Identified'
    peptides_identified = 'Peptide Sequences Identified' # 'peptides.txt' should have this number of peptides

df = mq_all_summaries.df
if df is not None:
    MS_spectra = df[[col_summary.MS1, col_summary.MS2, col_summary.MS2_identified, col_summary.peptides_identified]]

    def compute_summary(threshold_identified):
        mask  = MS_spectra[col_summary.peptides_identified] >= threshold_identified
        display(MS_spectra.loc[mask].describe(np.linspace(0.05, 0.95, 10)))
    
    w_ions_range = widgets.IntSlider(value=15_000, min=.0, max=MS_spectra[col_summary.peptides_identified].max())
    display(widgets.interactive(compute_summary, threshold_identified=w_ions_range))

# %%
mask  = MS_spectra[col_summary.peptides_identified] >= w_ions_range.value
logger.warning(f"Save {mask.sum()} file names to configuration file of selected samples: "
f"{ELIGABLE_FILES_YAML} "
f"based on  a minimum of {w_ions_range.value} peptides.")
idx_selected = MS_spectra.loc[mask].index
MS_spectra.loc[idx_selected]


# %% [markdown]
# ### Select Date Range
#
# - based on metadata

# %%
df_meta_rawfiles = pd.read_csv(FN_RAWFILE_METADATA, header=[0, 1], index_col=0)
date_col = ('FileProperties', 'Content Creation Date')
df_meta_rawfiles[date_col] = pd.to_datetime(
    df_meta_rawfiles[date_col])
df_meta_rawfiles = df_meta_rawfiles.loc[idx_selected]
df_meta_rawfiles.sort_values(date_col, inplace=True)

# %%
w_date_range = widgets.SelectionRangeSlider(options=df_meta_rawfiles[date_col], value=[min(df_meta_rawfiles[date_col]),max(df_meta_rawfiles[date_col]) ] )

def show(range):
    mask = df_meta_rawfiles[date_col].between(*range)
    df_view = MS_spectra.loc[idx_selected].loc[mask]
    display(df_view)


int_date_range = widgets.interactive(show, range=w_date_range)
display(int_date_range)

# %%
mask = df_meta_rawfiles[date_col].between(*w_date_range.value)
idx_selected = mask.loc[mask].index
idx_selected

# %% [markdown]
# ### Write out selected, eligable files

# %%
with open(ELIGABLE_FILES_YAML, 'w') as f:
    yaml.dump(data={'files': idx_selected.to_list()}, stream=f)
logger.info(f"Dumped yaml file with eligable files under key 'files' to {str(ELIGABLE_FILES_YAML)}")

# %% [markdown]
# ## Plot number of samples
#
# - binned by 10k steps

# %%
_max = MS_spectra[col_summary.peptides_identified].max() + 10_001
fig, ax = plt.subplots(figsize=(10,10))
_ = MS_spectra[col_summary.peptides_identified].hist(
    bins=range(0,_max, 10_000),
    legend=True,
    ax = ax)
fig.suptitle('Number of samples, binned in 10K steps.')
fig.tight_layout()

# %%
MS_spectra[col_summary.peptides_identified].mean(), MS_spectra[col_summary.peptides_identified].std() # including folders with 0 identified peptides


# %%
def calc_cutoff(threshold=1):
    s = MS_spectra[col_summary.peptides_identified]
    mask = s >= threshold
    s = s.loc[mask]
    display(f"Threshold selected (inclusive): {threshold} ")
    display(f"mean: {s.mean():.2f}, std-dev: {s.std():.2f}")


# calc_cutoff()
display(widgets.interactive(calc_cutoff, threshold=widgets.IntSlider(value=10000.0, min=.0, max=MS_spectra[col_summary.peptides_identified].max())))

# %%
fig, axes = plt.subplots(2,2, figsize=(20,20), sharex=True)

ylim_hist = (0,600)
xlim_dens = (0, 70_000)

ax = axes[0,0]
ax = mq_all_summaries.df[col_summary.peptides_identified].plot(kind='hist', bins=50, title="Histogram including samples with zero identified peptides", grid=True, ax=ax, ylim=ylim_hist)
ax = axes[1,0]
_ = mq_all_summaries.df[col_summary.peptides_identified].astype(float).plot.kde(ax=ax, title="Density plot including samples with zero identified peptides.", xlim=xlim_dens)

threshold_m2_identified = 15_000
mask = mq_all_summaries.df[col_summary.peptides_identified] >= threshold_m2_identified

ax = axes[0,1]
ax = mq_all_summaries.df.loc[mask, col_summary.peptides_identified].plot(kind='hist', bins=40, title=f"Histogram including samples with {threshold_m2_identified:,d} and more identified peptides", grid=True, ax=ax, ylim=ylim_hist)
ax = axes[1,1]
_ = mq_all_summaries.df.loc[mask, col_summary.peptides_identified].astype(float).plot.kde(ax=ax, title=f"Density plot including samples with {threshold_m2_identified:,d} and more identified peptides.", xlim=xlim_dens)

plotting._savefig(fig, name='distribution_peptides_in_samples', folder=config.FIGUREFOLDER)
