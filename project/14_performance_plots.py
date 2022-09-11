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
#     display_name: Python 3
#     language: python
#     name: python3
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
import seaborn as sns

pd.options.display.max_rows = 120
pd.options.display.min_rows = 50

import vaep
import vaep.imputation
from vaep import sampling
import vaep.models
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
# Parameters
fn_rawfile_metadata = "data/single_datasets/raw_meta.csv"
folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"

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
fig, axes = plt.subplots(1, 2, sharey=True)

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

# %%

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

imputed_shifted_normal = vaep.imputation.impute_shifted_normal(data.train_X, mean_shift=1.8, std_shrinkage=0.3)
imputed_shifted_normal

# %%
medians_train = data.train_X.median()
medians_train.name = 'median'

# %% [markdown]
# # Model specifications

# %%
import yaml 
def select_content(s:str, stub='metrics_'):
    s = s.split(stub)[1]
    assert isinstance(s, str), f"More than one split: {s}"
    entries = s.split('_')
    if len(entries) > 1:
        s = '_'.join(entries[:-1])
    return s

from functools import partial


all_configs = {}
for fname in args.out_models.iterdir():
    fname = Path(fname)
    if fname.suffix != '.yaml':
        continue
    # "grandparent" directory gives name beside name of file
    key = f"{select_content(fname.stem, 'config_')}"
    print(f"{key = }")
    with open(fname) as f:
        loaded = yaml.safe_load(f)   
    if key not in all_configs:
        all_configs[key] = loaded
        continue
    for k, v in loaded.items():
        if k in all_configs[key]:
            if not all_configs[key][k] == v:
                print(
                    "Diverging values for {k}: {v1} vs {v2}".format(
                k=k,
                v1=all_configs[key][k],
                v2=v)
                )
        else:
            all_configs[key][k] = v

model_configs = pd.DataFrame(all_configs).T
model_configs.T

# %% [markdown]
# # load predictions
#
# - calculate correlation -> only makes sense per feature (and than save overall correlation stats)

# %% [markdown]
# ## test data

# %%
split = 'test'
pred_files = [f for f in args.out_preds.iterdir() if split in f.name]
pred_test = compare_predictions.load_predictions(pred_files)
# pred_test = pred_test.join(medians_train, on=prop.index.name)
pred_test['random shifted normal'] = imputed_shifted_normal
pred_test = pred_test.join(freq_feat, on=freq_feat.index.name)
SAMPLE_ID, FEAT_NAME = pred_test.index.names
pred_test

# %%
pred_test_corr = pred_test.corr()
ax = pred_test_corr.loc['observed', ORDER_MODELS].plot.bar(
    title='Corr. between Fake NA and model predictions on test data',
    ylabel='correlation coefficient',
    ylim=(0.7,1)
)
ax = vaep.plotting.add_height_to_barplot(ax)
vaep.savefig(ax.get_figure(), name='pred_corr_test_overall', folder=args.out_figures)
pred_test_corr

# %%
corr_per_sample_test = pred_test.groupby('Sample ID').aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(ylim=(0.7,1), rot=90,
              title='Corr. betw. fake NA and model predictions per sample on test data',
              ylabel='correlation coefficient')
ax = corr_per_sample_test.plot.box(**kwargs)

vaep.savefig(ax.get_figure(), name='pred_corr_test_per_sample', folder=args.out_figures)
with pd.ExcelWriter(args.out_figures/'pred_corr_test_per_sample.xlsx') as writer:   
    corr_per_sample_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_test.to_excel(writer, sheet_name='correlations')

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
# ### Correlation per feature

# %%
corr_per_feat_test = pred_test.groupby(FEAT_NAME).aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(rot=90,
              title=f'Corr. per {FEAT_NAME} on test data',
              ylabel='correlation coefficient')
ax = corr_per_feat_test.plot.box(**kwargs)

vaep.savefig(ax.get_figure(), name='pred_corr_test_per_feat', folder=args.out_figures)
with pd.ExcelWriter(args.out_figures/'pred_corr_test_per_feat.xlsx') as writer:
    corr_per_feat_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_feat_test.to_excel(writer, sheet_name='correlations')

# %%
feat_count_test = data.test_y.stack().groupby(FEAT_NAME).count()
feat_count_test.name = 'count'
feat_count_test.head()

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_feat_test[models]).min()
mask = (corr_per_feat_test[models] < treshold).any(axis=1)

def highlight_min(s, color, tolerence=0.00001):
    return np.where((s - s.min()).abs() < tolerence, f"background-color: {color};", None)
corr_per_feat_test.join(feat_count_test).loc[mask].sort_values('count').style.apply(highlight_min, color='yellow', axis=1, subset=corr_per_feat_test.columns) 

# %%
metrics = vaep.models.Metrics(no_na_key='NA interpolated', with_na_key='NA not interpolated')
test_metrics = metrics.add_metrics(pred_test.drop('freq', axis=1), key='test data')
test_metrics = pd.DataFrame(test_metrics["NA interpolated"])[ORDER_MODELS]
test_metrics

# %%
n_in_comparison = int(test_metrics.loc['N'].unique()[0])
n_in_comparison

# %%
METRIC = 'MAE'
_to_plot = test_metrics.loc[METRIC].to_frame().T
_to_plot.index = [feature_names.name]
_to_plot

# %%
text = model_configs[["latent_dim", "hidden_layers"]].apply(
    lambda s: f'LD: {s["latent_dim"]:3} '
              f'- HL: {",".join(str(x) for x in s["hidden_layers"]) if s["hidden_layers"] is not np.nan else "-"}',
    axis=1)
text = text.rename({'dae': 'DAE', 'vae': 'VAE'})

_to_plot.loc["text"] = text
_to_plot = _to_plot.fillna('')
_to_plot

# %%
colors_to_use = [sns.color_palette()[5] ,*sns.color_palette()[:5]]
# list(sns.color_palette().as_hex()) # string representation of colors
sns.color_palette() # select colors for comparibility with grid search (where random shifted was omitted)

# %%
ax = _to_plot.loc[[feature_names.name]].plot.bar(rot=0,
                                                 ylabel=f"{METRIC} (log2 intensities)",
                                                 title=f'performance on test data (based on {n_in_comparison:,} measurements)',
                                                 color=colors_to_use,
                                                 width=.8)
ax = vaep.plotting.add_height_to_barplot(ax)
ax = vaep.plotting.add_text_to_barplot(ax, _to_plot.loc["text"], size=16)
fig = ax.get_figure()
vaep.savefig(fig, "performance_models_test", folder=args.out_figures)

# %%
errors_test = vaep.pandas.calc_errors_per_feat(pred_test.drop("freq", axis=1), freq_feat=freq_feat)[[*ORDER_MODELS, 'freq']]
errors_test


# %%
def plot_rolling_error(errors:pd.DataFrame, window:int=200, freq_col:str='freq', ax=None):
    errors_smoothed = errors.drop(freq_col, axis=1).rolling(window=window, min_periods=1).mean()
    errors_smoothed[freq_col] = errors[freq_col]
    ax = errors_smoothed.plot(x=freq_col, ylabel='rolling error average', color=colors_to_use, ax=None)
    return ax
ax = plot_rolling_error(errors_test, window=int(len(errors_test)/15))
vaep.savefig(ax.get_figure(), name='errors_rolling_avg_test',folder=args.out_figures)

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
vaep.savefig(ax.get_figure(), name='pred_corr_val_overall', folder=args.out_figures)
pred_val_corr

# %%
corr_per_sample_val = pred_val.groupby('Sample ID').aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(ylim=(0.7,1), rot=90,
              title='Corr. betw. fake NA and model pred. per sample on validation data',
              ylabel='correlation coefficient')
ax = corr_per_sample_val.plot.box(**kwargs)

vaep.savefig(ax.get_figure(), name='pred_corr_valid_per_sample', folder=args.out_figures)
with pd.ExcelWriter(args.out_figures/'pred_corr_valid_per_sample.xlsx') as writer:   
    corr_per_sample_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_test.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_sample_val[models]).min()
mask = (corr_per_sample_val[models] < treshold).any(axis=1)
corr_per_sample_val.loc[mask].style.highlight_min(axis=1)

# %% [markdown]
# ### Error plot

# %%
errors_val = pred_val.drop('observed', axis=1).sub(pred_val['observed'], axis=0)[ORDER_MODELS]
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
ax = errors_val_smoothed.plot(x=freq_feat.name, ylabel='rolling error average', color=colors_to_use)

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
