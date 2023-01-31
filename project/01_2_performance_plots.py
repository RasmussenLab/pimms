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
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

import vaep
import vaep.imputation
import vaep.models
from vaep.io import datasplits
from vaep.analyzers import compare_predictions
import vaep.nb

pd.options.display.max_rows = 120
pd.options.display.min_rows = 50
pd.options.display.max_colwidth = 100

logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
folder_experiment:str = 'runs/experiment_03/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070' # Datasplit folder with data for experiment
folder_data:str = '' # specify data directory if needed
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
models = 'CF,DAE,VAE'


# %% [markdown]
# Some argument transformations

# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
args = vaep.nb.args_from_dict(args)
args

# %%
figures = {}


# %%
MODELS = args.models.split(',')
ORDER_MODELS = ['RSN', 'median', 'interpolated', *MODELS]

# %%
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format)

# %%
fig, axes = plt.subplots(1, 2, sharey=True)

ax = axes[0]
_ = data.val_y.unstack().notna().sum(axis=1).sort_values().plot(
        ax=ax,
        title='Validation data',
        ylabel='number of feat')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax = axes[1]
_ = data.test_y.unstack().notna().sum(axis=1).sort_values().plot(
        ax=ax,
        title='Test data')
fig.suptitle("Fake NAs per sample availability.", size=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fname = args.out_figures / 'fake_na_val_test_splits.png'
figures[fname.stem] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ## Across data completeness

# %%
freq_feat = vaep.io.datasplits.load_freq(args.data, file='freq_features.json')   # needs to be pickle -> index.name needed

freq_feat.head() # training data

# %%
prop = freq_feat / len(data.train_X.index.levels[0])
prop.to_frame()

# %% [markdown]
# # reference methods
#
# - drawing from shifted normal distribution (RSN imputation)
# - median imputation

# %%
data.to_wide_format()
data.train_X

# %%
N_SAMPLES, M_FEAT = data.train_X.shape
print(f"N samples: {N_SAMPLES:,d}, M features: {M_FEAT}")

# %%
mean = data.train_X.mean()
std = data.train_X.std()

imputed_shifted_normal = vaep.imputation.impute_shifted_normal(data.train_X, mean_shift=1.8, std_shrinkage=0.3, axis=0)
imputed_shifted_normal.to_frame('intensity')

# %%
medians_train = data.train_X.median()
medians_train.name = 'median'

# %% [markdown]
# # Model specifications

# %%
import yaml
from vaep.models.collect_dumps import collect, select_content

def load_config_file(fname: Path, first_split='config_') -> dict:
    with open(fname) as f:
        loaded = yaml.safe_load(f)
    key = f"{select_content(fname.stem, first_split=first_split)}"
    return key, loaded

# model_key could be used as key from config file
# load only specified configs?
all_configs = collect(
    paths=(fname for fname in args.out_models.iterdir() if fname.suffix == '.yaml'),
    load_fn=load_config_file
)
model_configs = pd.DataFrame(all_configs).T
model_configs.T

# %% [markdown]
# # load predictions
#
# - calculate correlation -> only makes sense per feature (and than save overall correlation stats)

# %% [markdown]
# ## test data

# %%
# index name
freq_feat.index.name = data.train_X.columns.name

# %%
split = 'test'
pred_files = [f for f in args.out_preds.iterdir() if split in f.name]
pred_test = compare_predictions.load_predictions(pred_files)
# pred_test = pred_test.join(medians_train, on=prop.index.name) # ToDo: median implicit
pred_test['RSN'] = imputed_shifted_normal
pred_test = pred_test.join(freq_feat, on=freq_feat.index.name)
SAMPLE_ID, FEAT_NAME = pred_test.index.names
pred_test

# %%
pred_test_corr = pred_test.corr()
ax = pred_test_corr.loc['observed', ORDER_MODELS].plot.bar(
    # title='Corr. between Fake NA and model predictions on test data',
    ylabel='correlation coefficient overall',
    ylim=(0.7,1)
)
ax = vaep.plotting.add_height_to_barplot(ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
vaep.savefig(ax.get_figure(), name='pred_corr_test_overall', folder=args.out_figures)
pred_test_corr

# %%
# index name
sample_index_name = data.train_X.index.name

# %%
corr_per_sample_test = pred_test.groupby(sample_index_name).aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]
corr_per_sample_test = corr_per_sample_test.join(pred_test.groupby(sample_index_name)[
                                       'median'].count().rename('n_obs'))
too_few_obs = corr_per_sample_test['n_obs'] < 3
corr_per_sample_test.loc[~too_few_obs].describe()

# %%
kwargs = dict(ylim=(0.7,1), rot=90,
              # title='Corr. betw. fake NA and model predictions per sample on test data',
              ylabel='correlation per sample')
ax = corr_per_sample_test.plot.box(**kwargs)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fname = args.out_figures / 'pred_corr_test_per_sample.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
with pd.ExcelWriter(args.out_figures/'pred_corr_test_per_sample.xlsx') as writer:   
    corr_per_sample_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_test.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_sample_test[MODELS]).min()
mask = (corr_per_sample_test[MODELS] < treshold).any(axis=1)
corr_per_sample_test.loc[mask].style.highlight_min(axis=1)

# %%
feature_names = pred_test.index.levels[-1]
N_SAMPLES = pred_test.index
M = len(feature_names)
pred_test.loc[pd.IndexSlice[:, feature_names[random.randint(0, M)]], :]

# %%
options = random.sample(set(freq_feat.index), 1)
pred_test.loc[pd.IndexSlice[:, options[0]], :]

# %% [markdown]
# ### Correlation per feature

# %%
corr_per_feat_test = pred_test.groupby(FEAT_NAME).aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]
corr_per_feat_test = corr_per_feat_test.join(pred_test.groupby(FEAT_NAME)[
                                   'observed'].count().rename('n_obs'))

too_few_obs = corr_per_feat_test['n_obs'] < 3
corr_per_feat_test.loc[~too_few_obs].describe()

# %%
corr_per_feat_test.loc[too_few_obs].dropna(thresh=3, axis=0)

# %%
kwargs = dict(rot=90,
              # title=f'Corr. per {FEAT_NAME} on test data',
              ylabel=f'correlation per {FEAT_NAME}')
ax = corr_per_feat_test.loc[~too_few_obs].drop(
    'n_obs', axis=1).plot.box(**kwargs)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fname = args.out_figures / 'pred_corr_test_per_feat.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
with pd.ExcelWriter(args.out_figures/'pred_corr_test_per_feat.xlsx') as writer:
    corr_per_feat_test.loc[~too_few_obs].describe().to_excel(writer, sheet_name='summary')
    corr_per_feat_test.to_excel(writer, sheet_name='correlations')

# %%
feat_count_test = data.test_y.stack().groupby(FEAT_NAME).count()
feat_count_test.name = 'count'
feat_count_test.head()

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_feat_test[MODELS]).min()
mask = (corr_per_feat_test[MODELS] < treshold).any(axis=1)

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
sns.color_palette() # select colors for comparibility with grid search (where RSN was omitted)

# %%
fig, ax = plt.subplots(figsize=(10,8))
ax = _to_plot.loc[[feature_names.name]].plot.bar(rot=0,
                                                 ylabel=f"{METRIC} for {feature_names.name} (based on {n_in_comparison:,} log2 intensities)",
                                                 # title=f'performance on test data (based on {n_in_comparison:,} measurements)',
                                                 color=colors_to_use,
                                                 ax=ax,
                                                 width=.8)
ax = vaep.plotting.add_height_to_barplot(ax)
ax = vaep.plotting.add_text_to_barplot(ax, _to_plot.loc["text"], size=16)
ax.set_xticklabels([])
fname = args.out_figures / 'performance_test.pdf'
figures[fname.stem] = fname
vaep.savefig(fig, name=fname)

# %%
errors_test = vaep.pandas.calc_errors_per_feat(pred_test.drop("freq", axis=1), freq_feat=freq_feat)[[*ORDER_MODELS, 'freq']]
errors_test


# %%
min_freq = None
ax = vaep.plotting.plot_rolling_error(
    errors_test,
    metric_name=METRIC,
    window=int(len(errors_test)/15),
    min_freq=min_freq, 
    colors_to_use=colors_to_use)
fname = args.out_figures / 'errors_rolling_avg_test.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

# %% [markdown]
# ## Validation data

# %%
split = 'val'
pred_files = [f for f in args.out_preds.iterdir() if split in f.name]
pred_val = compare_predictions.load_predictions(pred_files)
# pred_val = pred_val.join(medians_train, on=freq_feat.index.name)
pred_val['RSN'] = imputed_shifted_normal
# pred_val = pred_val.join(freq_feat, on=freq_feat.index.name)
pred_val_corr = pred_val.corr()
ax = pred_val_corr.loc['observed', ORDER_MODELS].plot.bar(
    # title='Correlation between Fake NA and model predictions on validation data',
    ylabel='correlation overall')
ax = vaep.plotting.add_height_to_barplot(ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fname = args.out_figures / 'pred_corr_val_overall.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
pred_val_corr

# %%
corr_per_sample_val = pred_val.groupby(sample_index_name).aggregate(lambda df: df.corr().loc['observed'])[ORDER_MODELS]

kwargs = dict(ylim=(0.7,1), rot=90,
              # title='Corr. betw. fake NA and model pred. per sample on validation data',
              ylabel='correlation per sample')
ax = corr_per_sample_val.plot.box(**kwargs)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fname = args.out_figures / 'pred_corr_val_per_sample.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
with pd.ExcelWriter(args.out_figures/'pred_corr_valid_per_sample.xlsx') as writer:   
    corr_per_sample_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_test.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(corr_per_sample_val[MODELS]).min()
mask = (corr_per_sample_val[MODELS] < treshold).any(axis=1)
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
mask = (errors_val[MODELS].abs() > c_error_min).any(axis=1)
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
mask = (errors_val[MODELS] >= c_avg_error).any(axis=1)
errors_val.loc[mask]

# %%
errors_val_smoothed = errors_val.copy()
errors_val_smoothed[errors_val.columns[:-1]] = (errors_val
                                                [errors_val.columns[:-1]]
                                                .rolling(window=200, min_periods=1)
                                                .mean())
ax = vaep.plotting.plot_rolling_error(errors_test,
                                      metric_name=METRIC,
                                      window=int(len(errors_test)/15),
                                      min_freq=min_freq,
                                      colors_to_use=colors_to_use)

# %%
errors_val_smoothed.describe()

# %%
fname = args.out_figures / 'performance_methods_by_completness.pdf'
figures[fname.stem] = fname
vaep.savefig(
    ax.get_figure(),
    name=fname)

# %% [markdown]
# # Average errors per feature - example scatter for collab
# - see how smoothing is done, here `collab`
# - shows how features are distributed in training data

# %%
# scatter plots to see spread
model = MODELS[0]
ax = errors_val.plot.scatter(x=prop.name, y=model, c='darkblue', ylim=(0,2),
  # title=f"Average error per feature on validation data for {model}",
  ylabel=f'average error ({METRIC}) for {model} on valid. data')
fname = args.out_figures / 'performance_methods_by_completness_scatter.pdf'
figures[fname.stem] = fname
vaep.savefig(
    ax.get_figure(),
    name=fname
)

# %% [markdown]
# - [ ] plotly plot with number of observations the mean for each feature is based on

# %% [markdown]
# ## Figures dumped to disk
# %%
figures
