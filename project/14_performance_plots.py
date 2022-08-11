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
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Compare models

# %%
import logging
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


import vaep
import vaep.imputation
from vaep import sampling
from vaep.io import datasplits
from vaep.analyzers import compare_predictions


import vaep.nb
matplotlib.rcParams['figure.figsize'] = [10.0, 8.0]


logging.basicConfig(level=logging.INFO)

# %%
# files and folders
folder_experiment:str = 'runs/experiment_03/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070' # Datasplit folder with data for experiment
folder_data:str = '' # specify data directory if needed
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow

# %%
args = vaep.nb.Config()

args.fn_rawfile_metadata = fn_rawfile_metadata
del fn_rawfile_metadata

args.folder_experiment = Path(folder_experiment)
del folder_experiment
args.folder_experiment.mkdir(exist_ok=True, parents=True)

args.file_format = file_format
del file_format

args = vaep.nb.add_default_paths(args, folder_data=folder_data)
del folder_data

args

# %%
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format) 

# %%
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18,10))

_ = data.val_y.unstack().notna().sum(axis=1).sort_values().plot(
        rot=90,
        ax=axes[0],
        title='Validation data',
        ylabel='number of feat')
_ = data.test_y.unstack().notna().sum(axis=1).sort_values().plot(
        rot=90,
        ax=axes[1],
        title='Test data')

# %% [markdown]
# ## Across data completeness

# %%
freq_feat = sampling.frequency_by_index(data.train_X, 0)
freq_feat.name = 'freq'
freq_feat.head() # training data

# %%
prop = freq_feat / len(data.train_X.index.levels[0])
prop.to_frame()

# %% [markdown]
# ## reference methods
#
# - drawing from shifted normal distribution
# - drawing from (-) normal distribution?
# - median imputation

# %%
data.to_wide_format()
data.train_X

# %%
mean = data.train_X.mean()
std = data.train_X.std()

imputed_shifted_normal = vaep.imputation.impute_shifted_normal(data.train_X)
imputed_shifted_normal

# %%
imputed_normal = vaep.imputation.impute_shifted_normal(data.train_X, mean_shift=0.0, std_shrinkage=1.0)
imputed_normal

# %%
medians_train = data.train_X.median()
medians_train.name = 'median'

# %% [markdown]
# ## load predictions

# %%
split='test'
pred_files =  [f for f in args.out_preds.iterdir() if split in f.name]
pred_test = compare_predictions.load_predictions(pred_files)
# pred_test = pred_test.join(medians_train, on=prop.index.name)
pred_test['random shifted normal'] = imputed_shifted_normal
pred_test['random normal'] = imputed_normal
pred_test = pred_test.join(freq_feat, on=freq_feat.index.name)
pred_test

# %%
feature_names = pred_test.index.levels[-1]
M = len(feature_names)
pred_test.loc[pd.IndexSlice[:, feature_names[random.randint(0, M)]], :]

# %%
options = ['NCOR1', ]
pred_test.loc[pd.IndexSlice[:, options[0]], :]

# %%
split='val'
pred_files =  [f for f in args.out_preds.iterdir() if split in f.name]
pred_val = compare_predictions.load_predictions(pred_files)
# pred_val = pred_val.join(medians_train, on=freq_feat.index.name)
pred_val['random shifted normal'] = imputed_shifted_normal
# pred_val = pred_val.join(freq_feat, on=freq_feat.index.name)

# 
errors_val = pred_val.drop('observed', axis=1).sub(pred_val['observed'], axis=0)
errors_val = errors_val.abs().groupby(freq_feat.index.name).mean() # absolute error
errors_val = errors_val.join(freq_feat)
errors_val = errors_val.sort_values(by=freq_feat.name, ascending=True)
errors_val

# %% [markdown]
# Some interpolated features are missing

# %%
errors_val.describe() 

# %%
errors_val_smoothed = errors_val.copy()
errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x=freq_feat.name, ylabel='rolling error average')

# %%
vaep.savefig(
    ax.get_figure(),
    folder=args.out_figures,
    name='performance_methods_by_completness')

# %%
# scatter plots to see spread
errors_val.plot.scatter(x=prop.name, y='collab')
