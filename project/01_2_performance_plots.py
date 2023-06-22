# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Compare models
#
# 1. Load available configurations
# 2. Load validation predictions
#     - calculate absolute error
#     - select top N for plotting by MAE from smallest (best) to largest (worst) (top N as specified, default 5)
#     - correlation per sample, correlation per feat, correlation overall
#     - MAE plots
# 3. Load test data predictions
#     - as for validation data
#     - top N based on validation data

# %%
import yaml
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import vaep
import vaep.imputation
import vaep.models
from vaep.models.collect_dumps import collect, select_content
from vaep.io import datasplits
from vaep.analyzers import compare_predictions
import vaep.nb

pd.options.display.max_rows = 120
pd.options.display.min_rows = 50
pd.options.display.max_colwidth = 100

plt.rcParams.update({'figure.figsize': (4, 2)})
vaep.plotting.make_large_descriptors(5)

logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
# Datasplit folder with data for experiment
folder_experiment: str = 'runs/example'
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # change default to pickled files
# Machine parsed metadata from rawfile workflow
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
models: str = 'Median,CF,DAE,VAE'  # picked models to compare (comma separated)
# Restrict plotting to top N methods for imputation based on error of validation data, maximum 10
plot_to_n: int = 5

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
dumps = {}

# %%
TARGET_COL = 'observed'
METRIC = 'MAE'
MIN_FREQ = None
MODELS_PASSED = args.models.split(',')
MODELS = MODELS_PASSED.copy()

# MODELS = args.models.split(',')
# ORDER_MODELS = ['RSN', *MODELS]

# %%
# list(sns.color_palette().as_hex()) # string representation of colors
if args.plot_to_n > 10:
    logger.warning("Set maximum of models to 10 (maximum)")
    args.overwrite_entry('plot_to_n', 10)
COLORS_TO_USE = [sns.color_palette()[5], *sns.color_palette()[:5]]

# %%
vaep.plotting.defaults.assign_colors(['CF', 'DAE', 'knn', 'VAE'])

# %%
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)

# %%
fig, axes = plt.subplots(1, 2, sharey=True)

vaep.plotting.data.plot_observations(data.val_y.unstack(), ax=axes[0],
                                     title='Validation split', size=1)
vaep.plotting.data.plot_observations(data.test_y.unstack(), ax=axes[1],
                                     title='Test split', size=1)

fig.suptitle("Simulated missing values per sample", size=8)

fname = args.out_figures / 'fake_na_val_test_splits.png'
figures[fname.stem] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ## Across data completeness

# %%
# load frequency of training features...
# needs to be pickle -> index.name needed
freq_feat = vaep.io.datasplits.load_freq(args.data, file='freq_features.json')

freq_feat.head()  # training data

# %%
prop = freq_feat / len(data.train_X.index.levels[0])
prop.to_frame()

# %%
data.to_wide_format()
data.train_X

# %%
N_SAMPLES, M_FEAT = data.train_X.shape
print(f"N samples: {N_SAMPLES:,d}, M features: {M_FEAT}")

# %%
fname = args.folder_experiment / '01_2_performance_summary.xlsx'
dumps[fname.stem] = fname
writer = pd.ExcelWriter(fname)

# %% [markdown]
# # Model specifications
# - used for bar plot annotations

# %%


def load_config_file(fname: Path, first_split='config_') -> dict:
    with open(fname) as f:
        loaded = yaml.safe_load(f)
    key = f"{select_content(fname.stem, first_split=first_split)}"
    return key, loaded


# model_key could be used as key from config file
# load only specified configs?
# case: no config file available?
all_configs = collect(
    paths=(fname for fname in args.out_models.iterdir()
           if fname.suffix == '.yaml'
           and 'model_config' in fname.name),
    load_fn=load_config_file
)
model_configs = pd.DataFrame(all_configs).set_index('model')
model_configs.T.to_excel(writer, sheet_name='model_params')
model_configs.T

# %% [markdown]
# Set Feature name (columns are features, rows are samples)

# %%
# index name
freq_feat.index.name = data.train_X.columns.name

# %%
# index name
sample_index_name = data.train_X.index.name

# %% [markdown]
# # Load predictions on validation and test data split
#

# %% [markdown]
# ## Validation data
# - set top N models to plot based on validation data split

# %%
pred_val = compare_predictions.load_split_prediction_by_modelkey(
    experiment_folder=args.folder_experiment,
    split='val',
    model_keys=MODELS_PASSED,
    shared_columns=[TARGET_COL])
pred_val[MODELS]

# %%
errors_val = (pred_val
              .drop(TARGET_COL, axis=1)
              .sub(pred_val[TARGET_COL], axis=0)
              [MODELS])
errors_val.describe()  # over all samples, and all features

# %% [markdown]
# Describe absolute error

# %%
errors_val.abs().describe()  # over all samples, and all features

# %% [markdown]
# ## Select top N for plotting and set colors
# %%
ORDER_MODELS = (errors_val
                .abs()
                .mean()
                .sort_values()
                .index
                .to_list())
ORDER_MODELS

# %%
mae_stats_ordered = errors_val.abs().describe()[ORDER_MODELS]
mae_stats_ordered.to_excel(writer, sheet_name='mae_stats_ordered')
writer.close()
mae_stats_ordered

# %% [markdown]
# Hack color order, by assing CF, DAE and VAE unique colors no matter their order
# Could be extended to all supported imputation methods
# %%
COLORS_TO_USE = vaep.plotting.defaults.assign_colors(ORDER_MODELS)

# %%
# For top_N -> define colors
TOP_N_ORDER = ORDER_MODELS[:args.plot_to_n]

TOP_N_COLOR_PALETTE = {model: color for model,
                       color in zip(TOP_N_ORDER, COLORS_TO_USE)}

TOP_N_ORDER

# %% [markdown]
# ### Correlation overall

# %%
pred_val_corr = pred_val.corr()
ax = (pred_val_corr
      .loc[TARGET_COL, ORDER_MODELS]
      .plot
      .bar(
          # title='Correlation between Fake NA and model predictions on validation data',
          ylabel='correlation overall'))
ax = vaep.plotting.add_height_to_barplot(ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
fname = args.out_figures / 'pred_corr_val_overall.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
pred_val_corr

# %% [markdown]
# ### Correlation per sample

# %%
corr_per_sample_val = (pred_val
                       .groupby(sample_index_name)
                       .aggregate(
                           lambda df: df.corr().loc[TARGET_COL]
                       )[ORDER_MODELS])

kwargs = dict(ylim=(0.7, 1), rot=90,
              #     boxprops=dict(linewidth=1.5),
              flierprops=dict(markersize=3),
              # title='Corr. betw. fake NA and model pred. per sample on validation data',
              ylabel='correlation per sample')
ax = corr_per_sample_val[TOP_N_ORDER].plot.box(**kwargs)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
fname = args.out_figures / 'pred_corr_val_per_sample.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

fname = args.out_figures/'pred_corr_val_per_sample.xlsx'
dumps[fname.stem] = fname
with pd.ExcelWriter(fname) as writer:
    corr_per_sample_val.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_val.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(
    corr_per_sample_val[TOP_N_ORDER]).min()
mask = (corr_per_sample_val[TOP_N_ORDER] < treshold).any(axis=1)
corr_per_sample_val.loc[mask].style.highlight_min(
    axis=1) if mask.sum() else 'Nothing to display'

# %% [markdown]
# ### Error plot

# %%
c_error_min = 4.5
mask = (errors_val[MODELS].abs() > c_error_min).any(axis=1)
errors_val.loc[mask].sort_index(level=1)

# %%
errors_val = errors_val.abs().groupby(
    freq_feat.index.name).mean()  # absolute error
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
ax = vaep.plotting.plot_rolling_error(errors_val[TOP_N_ORDER + ['freq']],
                                      metric_name=METRIC,
                                      window=int(len(errors_val)/15),
                                      min_freq=MIN_FREQ,
                                      colors_to_use=COLORS_TO_USE)
fname = args.out_figures / 'performance_methods_by_completness_val.pdf'
figures[fname.stem] = fname
vaep.savefig(
    ax.get_figure(),
    name=fname)


# %% [markdown]
# ### Error by non-decimal number of intensity
# - number of observations in parentheses.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax, errors_binned = vaep.plotting.errors.plot_errors_binned(
    pred_val[
        [TARGET_COL]+TOP_N_ORDER
    ],
    ax=ax,
    palette=TOP_N_COLOR_PALETTE,
    metric_name=METRIC,)
ax.set_ylabel(f"Average error ({METRIC})")
fname = args.out_figures / 'errors_binned_by_int_val.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

# %%
errors_binned.head()
dumps[fname.stem] = fname.with_suffix('.csv')
errors_binned.to_csv(fname.with_suffix('.csv'))
errors_binned.head()

# %% [markdown]
# ## test data

# %%
pred_test = compare_predictions.load_split_prediction_by_modelkey(
    experiment_folder=args.folder_experiment,
    split='test',
    model_keys=MODELS_PASSED,
    shared_columns=[TARGET_COL])
pred_test = pred_test.join(freq_feat, on=freq_feat.index.name)
SAMPLE_ID, FEAT_NAME = pred_test.index.names
pred_test

# %% [markdown]
# ## Intensity distribution as histogram
# plot top 4 models
# %%
min_max = vaep.plotting.data.min_max(pred_test[TARGET_COL])
top_n =  4
fig, axes = plt.subplots(ncols=4, figsize=(8, 2), sharey=True)

for model, color, ax in zip(
    ORDER_MODELS[:4],
    COLORS_TO_USE[:4],
    axes):
    
    ax, _ = vaep.plotting.data.plot_histogram_intensites(
        pred_test[TARGET_COL],
        color='grey',
        min_max=min_max,
        ax=ax
    )
    ax, _ = vaep.plotting.data.plot_histogram_intensites(
        pred_test[model],
        color=color,
        min_max=min_max,
        ax=ax,
        alpha=0.5,
    )
    _ = [(l.set_rotation(90))
             for l in ax.get_xticklabels()]
    ax.legend()

axes[0].set_ylabel('Number of observations')

fname = args.out_figures / f'intensity_binned_top_{top_n}_models_test.pdf'
figures[fname.stem] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ### Correlation overall

# %%
pred_test_corr = pred_test.corr()
ax = pred_test_corr.loc[TARGET_COL, ORDER_MODELS].plot.bar(
    # title='Corr. between Fake NA and model predictions on test data',
    ylabel='correlation coefficient overall',
    ylim=(0.7, 1)
)
ax = vaep.plotting.add_height_to_barplot(ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
fname = args.out_figures / 'pred_corr_test_overall.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
pred_test_corr

# %% [markdown]
# ### Correlation per sample

# %%
corr_per_sample_test = (pred_test
                        .groupby(sample_index_name)
                        .aggregate(lambda df: df.corr().loc[TARGET_COL])
                        [ORDER_MODELS])
corr_per_sample_test = corr_per_sample_test.join(
    pred_test
    .groupby(sample_index_name)[TARGET_COL]
    .count()
    .rename('n_obs')
)
too_few_obs = corr_per_sample_test['n_obs'] < 3
corr_per_sample_test.loc[~too_few_obs].describe()

# %%
kwargs = dict(ylim=(0.7, 1), rot=90,
              flierprops=dict(markersize=3),
              # title='Corr. betw. fake NA and model predictions per sample on test data',
              ylabel='correlation per sample')
ax = (corr_per_sample_test
      .loc[~too_few_obs, TOP_N_ORDER]
      .plot
      .box(**kwargs))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
fname = args.out_figures / 'pred_corr_test_per_sample.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

dumps[fname.stem] = fname.with_suffix('.xlsx')
with pd.ExcelWriter(fname.with_suffix('.xlsx')) as writer:
    corr_per_sample_test.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample_test.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# identify samples which are below lower whisker for models

# %%
treshold = vaep.pandas.get_lower_whiskers(
    corr_per_sample_test[TOP_N_ORDER]).min()
mask = (corr_per_sample_test[TOP_N_ORDER] < treshold).any(axis=1)
corr_per_sample_test.loc[mask].style.highlight_min(
    axis=1) if mask.sum() else 'Nothing to display'

# %%
feature_names = pred_test.index.levels[-1]
N_SAMPLES = pred_test.index
M = len(feature_names)
pred_test.loc[pd.IndexSlice[:, feature_names[random.randint(0, M)]], :]

# %%
options = random.sample(set(feature_names), 1)
pred_test.loc[pd.IndexSlice[:, options[0]], :]

# %% [markdown]
# ### Correlation per feature

# %%
corr_per_feat_test = pred_test.groupby(FEAT_NAME).aggregate(
    lambda df: df.corr().loc[TARGET_COL])[ORDER_MODELS]
corr_per_feat_test = corr_per_feat_test.join(pred_test.groupby(FEAT_NAME)[
    TARGET_COL].count().rename('n_obs'))

too_few_obs = corr_per_feat_test['n_obs'] < 3
corr_per_feat_test.loc[~too_few_obs].describe()

# %%
corr_per_feat_test.loc[too_few_obs].dropna(thresh=3, axis=0)

# %%
kwargs = dict(rot=90,
              flierprops=dict(markersize=1),
              ylabel=f'correlation per {FEAT_NAME}')
ax = (corr_per_feat_test
      .loc[~too_few_obs, TOP_N_ORDER]
      .plot
      .box(**kwargs)
      )
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
fname = args.out_figures / 'pred_corr_test_per_feat.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)
dumps[fname.stem] = fname.with_suffix('.xlsx')
with pd.ExcelWriter(fname.with_suffix('.xlsx')) as writer:
    corr_per_feat_test.loc[~too_few_obs].describe().to_excel(
        writer, sheet_name='summary')
    corr_per_feat_test.to_excel(writer, sheet_name='correlations')

# %%
feat_count_test = data.test_y.stack().groupby(FEAT_NAME).count()
feat_count_test.name = 'count'
feat_count_test.head()

# %%
treshold = vaep.pandas.get_lower_whiskers(
    corr_per_feat_test[TOP_N_ORDER]).min()
mask = (corr_per_feat_test[TOP_N_ORDER] < treshold).any(axis=1)


def highlight_min(s, color, tolerence=0.00001):
    return np.where((s - s.min()).abs() < tolerence, f"background-color: {color};", None)


view = (corr_per_feat_test
        .join(feat_count_test)
        .loc[mask]
        .sort_values('count'))

if not view.empty:
    display(view
            .style.
            apply(highlight_min, color='yellow', axis=1,
                  subset=corr_per_feat_test.columns)
            )
else:
    print("None found")
# %% [markdown]
# ### Error plot

# %%
metrics = vaep.models.Metrics()
test_metrics = metrics.add_metrics(
    pred_test.drop('freq', axis=1), key='test data')
test_metrics = pd.DataFrame(test_metrics)[TOP_N_ORDER]
test_metrics

# %%
n_in_comparison = int(test_metrics.loc['N'].unique()[0])
n_in_comparison

# %%
_to_plot = test_metrics.loc[METRIC].to_frame().T
_to_plot.index = [feature_names.name]
_to_plot

# %%


def build_text(s):
    ret = ''
    if not np.isnan(s["latent_dim"]):
        ret += f'LD: {int(s["latent_dim"])} '
    if not np.isnan(s["hidden_layers"]):
        t = ",".join(str(x) for x in s["hidden_layers"])
        ret += f"HL: {t}"
    return ret


text = model_configs[["latent_dim", "hidden_layers"]].apply(
    build_text,
    axis=1)

_to_plot.loc["text"] = text
_to_plot = _to_plot.fillna('')
_to_plot


# %%
fig, ax = plt.subplots(figsize=(4, 2))
ax = _to_plot.loc[[feature_names.name]].plot.bar(rot=0,
                                                 ylabel=f"{METRIC} for {feature_names.name} (based on {n_in_comparison:,} log2 intensities)",
                                                 # title=f'performance on test data (based on {n_in_comparison:,} measurements)',
                                                 color=COLORS_TO_USE,
                                                 ax=ax,
                                                 width=.8)
ax = vaep.plotting.add_height_to_barplot(ax, size=5)
ax = vaep.plotting.add_text_to_barplot(ax, _to_plot.loc["text"], size=5)
ax.set_xticklabels([])
fname = args.out_figures / 'performance_test.pdf'
figures[fname.stem] = fname
vaep.savefig(fig, name=fname)

# %%
dumps[fname.stem] = fname.with_suffix('.csv')
_to_plot_long = _to_plot.T
_to_plot_long = _to_plot_long.rename(
    {feature_names.name: 'metric_value'}, axis=1)
_to_plot_long['data level'] = feature_names.name
_to_plot_long = _to_plot_long.set_index('data level', append=True)
_to_plot_long.to_csv(fname.with_suffix('.csv'))

# %%
errors_test = vaep.pandas.calc_errors_per_feat(pred_test.drop(
    "freq", axis=1), freq_feat=freq_feat)[[*TOP_N_ORDER, 'freq']]
errors_test
# %% [markdown]
# ### Error plot by frequency

# %%
ax = vaep.plotting.plot_rolling_error(
    errors_test,
    metric_name=METRIC,
    window=int(len(errors_test)/15),
    min_freq=MIN_FREQ,
    colors_to_use=COLORS_TO_USE)
fname = args.out_figures / 'errors_rolling_avg_test.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)


# %% [markdown]
# Plot error by median feature intensity

# %%
fig, ax = plt.subplots(figsize=(8,2))

ax, errors_binned = vaep.plotting.errors.plot_errors_by_median(
    pred=pred_test[
        [TARGET_COL]+TOP_N_ORDER
    ],
    feat_medians=data.train_X.median(),
    ax=ax,
    metric_name=METRIC,
    palette=COLORS_TO_USE
)

fname = args.out_figures / 'errors_binned_by_feat_medians.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

dumps[fname.stem] = fname.with_suffix('.csv')
errors_binned.to_csv(fname.with_suffix('.csv'))
errors_binned

# %%
(errors_binned
 .set_index(
     ['model', errors_binned.columns[-1]]
 )
 .loc[ORDER_MODELS[0]]
 .sort_values(by=METRIC))

# %% [markdown]
# ### Error by non-decimal number of intensity
#
# - number of observations in parentheses.

# %%
fig, ax = plt.subplots(figsize=(8, 2))
ax, errors_binned = vaep.plotting.errors.plot_errors_binned(
    pred_test[
        [TARGET_COL]+TOP_N_ORDER
    ],
    ax=ax,
    palette=TOP_N_COLOR_PALETTE,
    metric_name=METRIC,
)
fname = args.out_figures / 'errors_binned_by_int_test.pdf'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), name=fname)

# %%
dumps[fname.stem] = fname.with_suffix('.csv')
errors_binned.to_csv(fname.with_suffix('.csv'))
errors_binned.head()
# %% [markdown]
# ## Figures dumped to disk
# %%
figures
# %%
dumps

# %%
