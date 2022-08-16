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

pd.options.display.max_rows = 120
pd.options.display.min_rows = 50

import vaep
import vaep.imputation
from vaep import sampling
from vaep.io import datasplits
from vaep.analyzers import compare_predictions


import vaep.nb
matplotlib.rcParams['figure.figsize'] = [10.0, 8.0]


logging.basicConfig(level=logging.INFO)

# %%
models = ['collab', 'DAE', 'VAE']
ORDER_MODELS = ['random shifted normal', 'median', 'interpolated',
                'collab', 'DAE', 'VAE',
                ]

# %% tags=["parameters"]
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
fig.suptitle("Fake NAs per sample availability.", size=24)
fig.tight_layout()
vaep.savefig(fig, name='fake_na_val_test_splits', folder=args.out_figures)

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
# # reference methods
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
# # load predictions

# %% [markdown]
# ## test data

# %%
split = 'test'
pred_files = [f for f in args.out_preds.iterdir() if split in f.name]
pred_test = compare_predictions.load_predictions(pred_files)
# pred_test = pred_test.join(medians_train, on=prop.index.name)
pred_test['random shifted normal'] = imputed_shifted_normal
pred_test['random normal'] = imputed_normal
pred_test = pred_test.join(freq_feat, on=freq_feat.index.name)
pred_test_corr = pred_test.corr()
ax = pred_test_corr.loc['observed', ORDER_MODELS].plot.bar(
    title='Corr. between Fake NA and model predictions on test data',
    ylabel='correlation coefficient',
    ylim=(0.7,1)
)
ax = vaep.plotting.add_height_to_barplot(ax)
vaep.savefig(ax.get_figure(), name='pred_corr_test', folder=args.out_figures)
pred_test_corr

# %%
corr_per_sample_test = pred_test.groupby('Sample ID').aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(ylim=(0.7,1), rot=90,
              title='Corr. betw. fake NA and model predictions per sample on test data',
              ylabel='correlation coefficient')
ax = corr_per_sample_test.plot.box(**kwargs)
fig = ax.get_figure()
fig.tight_layout()

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_sample_test[models]).min()
mask = (corr_per_sample_test[models] < treshold).any(axis=1)
corr_per_sample_test.loc[mask].style.highlight_min(axis=1)

# %%
feature_names = pred_test.index.levels[-1]
M = len(feature_names)
pred_test.loc[pd.IndexSlice[:, feature_names[random.randint(0, M)]], :]

# %%
options = random.sample(set(freq_feat.index), 1)
pred_test.loc[pd.IndexSlice[:, options[0]], :]

# %% [markdown]
# ## Validation data

# %%
split = 'val'
pred_files = [f for f in args.out_preds.iterdir() if split in f.name]
pred_val = compare_predictions.load_predictions(pred_files)
# pred_val = pred_val.join(medians_train, on=freq_feat.index.name)
pred_val['random shifted normal'] = imputed_shifted_normal
# pred_val = pred_val.join(freq_feat, on=freq_feat.index.name)
pred_val_corr = pred_val.corr()
ax = pred_val_corr.loc['observed', ORDER_MODELS].plot.bar(
    title='Correlation between Fake NA and model predictions on validation data',
    ylabel='correlation coefficient')
ax = vaep.plotting.add_height_to_barplot(ax)
vaep.savefig(ax.get_figure(), name='pred_corr_val', folder=args.out_figures)
pred_val_corr

# %%
corr_per_sample_val = pred_val.groupby('Sample ID').aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(ylim=(0.7,1), rot=90,
              title='Corr. betw. fake NA and model pred. per sample on validation data',
              ylabel='correlation coefficient')
ax = corr_per_sample_val.plot.box(**kwargs)
fig = ax.get_figure()
fig.tight_layout()

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_sample_val[models]).min()
mask = (corr_per_sample_val[models] < treshold).any(axis=1)
corr_per_sample_val.loc[mask].style.highlight_min(axis=1)

# %%
errors_val = pred_val.drop('observed', axis=1).sub(pred_val['observed'], axis=0)
errors_val.describe() # over all samples, and all features

# %% [markdown]
# Describe absolute error

# %%
errors_val.abs().describe() # over all samples, and all features

# %%
c_error_min = 4.5
mask = (errors_val[models].abs() > c_error_min).any(axis=1)
errors_val.loc[mask].sort_index(level=1)

# %%
errors_val = errors_val.abs().groupby(freq_feat.index.name).mean() # absolute error
errors_val = errors_val.join(freq_feat)
errors_val = errors_val.sort_values(by=freq_feat.name, ascending=True)
errors_val

# %% [markdown]
# Some interpolated features are missing

# %%
errors_val.describe()  # mean of means

# %%
c_avg_error = 2
mask = (errors_val[models] >= c_avg_error).any(axis=1)
errors_val.loc[mask]

# %%
errors_val_smoothed = errors_val.copy()
errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x=freq_feat.name, ylabel='rolling error average', ylim=(0,2))

# %%
errors_val_smoothed.describe()

# %%
vaep.savefig(
    ax.get_figure(),
    folder=args.out_figures,
    name='performance_methods_by_completness')

# %% [markdown]
# # Average errors per feature - example scatter for collab
# - see how smoothing is done, here `collab`
# - shows how features are distributed in training data

# %%
# scatter plots to see spread
model = models[0]
ax = errors_val.plot.scatter(x=prop.name, y=model, c='darkblue', ylim=(0,2),
  title=f"Average error per feature on validation data for {model}",
  ylabel='absolute error')

vaep.savefig(
    ax.get_figure(),
    folder=args.out_figures,
    name='performance_methods_by_completness_scatter',
)

# %% [markdown]
# - [ ] plotly plot with number of observations the mean for each feature is based on
# - [ ] 
