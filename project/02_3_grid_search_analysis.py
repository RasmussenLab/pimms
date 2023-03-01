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
# # Analyis of grid hyperparameter search

# %%
import json
import pathlib
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt


import vaep.nb
matplotlib.rcParams['figure.figsize'] = [12.0, 6.0]

import vaep.io
import vaep.pandas
import vaep.utils
from vaep.io import datasplits
from vaep import sampling
from vaep.analyzers import compare_predictions
import vaep.plotting.plotly as px_vaep

pd.options.display.max_columns = 45
pd.options.display.max_rows = 100
pd.options.display.multi_sparse = False

logger = vaep.logging.setup_nb_logger()

# %% [markdown]
# ## Papermill parameters

# %% [markdown]
# papermill parameters:

# %% tags=["parameters"]
metrics_csv:str = "path/to/all_metrics.csv" # file path to metrics
configs_csv:str = "path/to/all_configs.csv" # file path to configs ("meta data")

# %%
try:
    assert pathlib.Path(metrics_csv).exists()
    assert pathlib.Path(configs_csv).exists()
except AssertionError:
    metrics_csv = snakemake.input.metrics
    configs_csv = snakemake.input.config
    print(f"{metrics_csv = }", f"{configs_csv = }", sep="\n")

# %%
path_metrics = pathlib.Path(metrics_csv)
path_configs = pathlib.Path(configs_csv)
FOLDER = path_metrics.parent

# %%
files_out = dict()


# %% [markdown]
# ## Metrics of each run
#
# Metrics a `pandas.DataFrame`:

# %%
metrics = pd.read_csv(path_metrics, index_col=0, header=[0, 1, 2, 3])
metrics.head()

# %%
metrics.stack('model')

# %%
# ToDo: integrate as parameters
metric_columns = ['MSE', 'MAE']
model_keys = metrics.stack('model').index.levels[-1].unique().to_list() # not used
subset = metrics.columns.levels[0][0]
print(f"{subset = }")

# %% [markdown]
# ## Configuration of each run

# %% [markdown]
# Experiment metadata from configs

# %%
meta = pd.read_csv(path_configs)
meta['hidden_layers'] = (meta
                         .loc[meta['hidden_layers'].notna(), 'hidden_layers']
                         .apply(lambda x: tuple(eval(x)))
)
meta['n_hidden_layers'] = (meta
                           .loc[meta['hidden_layers'].notna(), 'hidden_layers']
                           .apply(len)
)
meta['n_hidden_layers'] = (meta
                           ['n_hidden_layers']
                           .fillna(0)
                           .astype(int)
)
meta.loc[meta['hidden_layers'].isna(), 'hidden_layers'] = None
meta = meta.set_index('id')
meta

# %% [markdown]
# Batch size for collab models depends on a factor (as the data in long format has roughly  N samples * M features entries).

# %% [markdown]
# ## Colorcoded metrics
#
# - can be one of the [matplotlib color maps](https://matplotlib.org/stable/tutorials/colors/colormaps.html), which also have reversed version indicated by `*_r`

# %%
sel_meta = (meta
            .loc[metrics.index, ['latent_dim', 'hidden_layers', 'batch_size']]
            )
sel_meta

# %%
cmap = 'cividis_r'

# %%
# ToDo: To make it cleaner: own config for each model (interpolated and median)
metrics_styled = (metrics
                 .set_index(
                     pd.MultiIndex
                     .from_frame(
                         meta
                        .loc[metrics.index, ['latent_dim', 'hidden_layers', 'batch_size']]
                        # .loc[metrics.index]
                     )
                 )
                .sort_index()
                .stack('model')
                .drop_duplicates()
                .style.background_gradient(cmap)
)

metrics = metrics_styled.data
metrics_styled

# %%
fname = FOLDER / 'metrics_styled.xlsx'
files_out['metrics_styled.xlsx'] = fname
metrics_styled.to_excel(fname)
logger.info(f"Saved styled metrics: {fname}")

# %% [markdown]
# ## Plot Top 10 for simulated Na validation data

# %%
ax = metrics[subset]["valid_fake_na"].sort_values(
    'MSE').iloc[:10, :-1].plot(rot=45,
                               x_compat=False,
                               xticks=list(range(10)),
                               marker='o',
                               linestyle='',
                               )
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='top_10_models_validation_fake_na', folder=FOLDER)

# %% [markdown]
# ## Create metrics in long format
#
# To use colors meaningfully, the long format of the data is needed.
#
# Rebuild metrics from dictionary

# %%
metrics_long = pd.read_csv(path_metrics, index_col=[0], header=[0,1,2,3])
# columns_names = ['subset', 'data_split', 'model', 'metric_name']
columns_names = list(metrics_long.columns.names)
metrics_long.sample(5) if len(metrics_long) > 15 else metrics_long

# %% [markdown]
# Combine with total number of simulated NAs the metric is based on (`N`) into single column

# %%
metrics_N = (metrics_long
             .loc[:, pd.IndexSlice[:, :, :, 'N']]
             .stack(['subset', 'data_split', 'model'])
             .reset_index()
             .drop_duplicates()
             .set_index(['id', 'subset', 'data_split', 'model'])
             .astype(int)
)
metrics_N

# %% [markdown]
# join total number of simulated NAs (`N`) used to compute metric

# %%
metrics_long = (metrics_long
                .loc[:, pd.IndexSlice[:, :, :, metric_columns]]
                .stack(metrics_long.columns.names)
                .to_frame('metric_value')
                .reset_index('metric_name')
                .join(metrics_N)
)
metrics_long

# %% [markdown]
# join metadata for each metric

# %%
metrics_long = (metrics_long
                .reset_index(['subset', 'data_split'])
                .join(meta.set_index('model', append=True))
               ).reset_index('model')
# metrics_long.index.name = 'id'
metrics_long.sample(5)

# %% [markdown]
# Combine number of parameters into one columns (they are mutually exclusive)

# %%
# ToDo: Still hacky: every model needs a config file (add prop. imputed)
# groupby 'id'
cols = ['M', 'data', 'file_format', 'fn_rawfile_metadata',
        'folder_data', 'folder_experiment',
        'level', 'meta_cat_col', 'meta_date_col', 
        'out_figures', 'out_folder', 'out_metrics', 'out_models', 'out_preds',
        'sample_idx_position', 'save_pred_real_na']
metrics_long[cols] = metrics_long.groupby(level=0)[cols].fillna(method='pad')
metrics_long.sample(5)

# %%
mask = metrics_long.model == 'interpolated'
# at least overall (and 1 for the number of replicates?)
metrics_long.loc[mask, 'n_params'] = 1
mask = metrics_long.model == 'Median'
# number of features to calculate median of
metrics_long.loc[mask, 'n_params'] = metrics_long.loc[mask, 'M']

metrics_long[[*columns_names, 'n_params',
             #'n_params_vae', 'n_params_dae', 'n_params_collab'
              ]]

# %% [markdown]
# A a descriptive column describing the `subset` and the total number of simulated NAs in it.

# %%
metrics_long['subset_w_N'] = metrics_long['subset'].str[0:] + \
    ' - N: ' + metrics_long['N'].apply(lambda x: f"{x:,d}")
metrics_long[['subset_w_N', 'subset']]

# %% [markdown]
# Save for later inspection

# %%
fname = FOLDER / 'metrics_long_df.csv'
files_out[fname.stem] = fname
metrics_long.to_csv(fname)  # Should all the plots be done without the metrics?
logger.info(f"Saved metrics in long format: {fname}")

# %% [markdown]
# # Collection of Performance plots 
#
# - specify `labels_dict` for plotly plotting
#
#

# %%
labels_dict = {"NA not interpolated valid_collab collab MSE": 'MSE',
               'batch_size': 'bs',
               'n_hidden_layers': "No. of hidden layers",
               'latent_dim': 'hidden layer dimension',
               'subset_w_N': 'subset',
               'n_params': 'no. of parameter',
               "metric_value": 'value',
               'metric_name': 'metric',
               'freq': 'freq/feature prevalence (across samples)'}

# %% [markdown]
# ## Plot result of hyperparameter search for collaborative Filtering

# %%
# not robust
category_orders = {'model': ['median', 'interpolated', 'CF', 'DAE', 'VAE'],
                   }

# %%
col = "NA interpolated valid_fake_na collab MAE"
model = 'CF'
# col = ("NA interpolated","valid_fake_na",'CF',"MSE")
fig = px.scatter(metrics_long.query(f'model == "{model}"'),
                 x="latent_dim",
                 y='metric_value',
                 color="subset",  # needs data in long format
                 facet_row="metric_name",
                 facet_col="data_split",
                 title='Performance of collaborative filtering models',
                 labels={**labels_dict, 'data_split': 'data split'},
                 category_orders={'data_split': [
                     'valid_fake_na', 'test_fake_na']},
                 width=1600,
                 height=700,
                 template='none',
                 )
fig.update_layout(
    font={'size': 18},
    xaxis={'title': {'standoff': 15}},
    yaxis={'title': {'standoff': 15}})
fig.update_xaxes(dict(
    tickmode='array',
    tickvals=sorted(metrics_long["latent_dim"].unique()),
)
)
fname = FOLDER / 'collab_performance_overview.pdf'
files_out['collab_performance_overview.pdf'] = fname
fig.write_image(fname)
logger.info(fname)
fig.show()

# %% [markdown]
# ## Plot hyperparameter search results - overview

# %%
hover_data = {k: ':,d' for k in
              ['hidden_layers',
               'latent_dim', 'n_params',
               'batch_size', 'N'
               ]}
hover_data['data_split'] = True
hover_data['metric_value'] = ':.4f'


def plot_by_params(data_split: str = '', subset: str = ''):
    selected = metrics_long
    if data_split:
        selected = selected.query(f'data_split == "{data_split}"')
    if subset:
        selected = selected.query(f'subset == "{subset}"')
    fig = px.scatter(selected,
                     x='n_params',
                     y='metric_value',
                     color="model",
                     facet_row="metric_name",
                     facet_col="subset_w_N",
                     hover_data=hover_data,
                     title=f'Performance by number of parameters for {data_split.replace("_", " ")} data'.replace(
                         "  ", " "),
                     labels=labels_dict,
                     category_orders=category_orders,
                     width=1600,
                     height=700,
                     template='none',
                     )

    fig.update_layout(
        font={'size': 18},
        xaxis={'title': {'standoff': 15}},
        yaxis={'title': {'standoff': 15}})
    return fig


dataset = "valid_fake_na"
fig = plot_by_params(dataset)
files_out[f"hyperpar_{dataset}_results_by_parameters_all.pdf"] = (FOLDER /
                                                                  f"hyperpar_{dataset}_results_by_parameters_all.pdf")
fig.write_image(files_out[f"hyperpar_{dataset}_results_by_parameters_all.pdf"])
fig

# %% [markdown]
# Only for NA which could also be interpolated for both test and validation data

# %%
fig = plot_by_params('', subset='NA interpolated')
fname = FOLDER / f"hyperpar_test_and_valid_fake_na_by_parameters_na_interpolated.pdf"
files_out[f"hyperpar_test_and_valid_fake_na_by_parameters_na_interpolated.pdf"] = fname
fig.write_image(fname)
logger.info(fname)
fig

# %% [markdown]
# Only for NA which could also be interpolated for both test and validation data

# %%
dataset = "test_fake_na"
fig = plot_by_params(dataset, 'NA interpolated')
fname = (FOLDER /
         f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf")
files_out[f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf"] = fname
fig.write_image(fname)
logger.info(f"Save to {fname}")
fig

# %%
dataset = "valid_fake_na"
fig = plot_by_params(dataset, 'NA interpolated')
fname = (FOLDER /
         f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf")
files_out[f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf"] = fname
fig.write_image(fname)
logger.info(f"Save to {fname}")
fig

# %% [markdown]
# ## Select best model for each `latent_dim`

# %%
group_by = ['data_split', 'subset', 'latent_dim', 'metric_name', 'model']
metrics_long_sel_min = metrics_long.reset_index(
        ).groupby(by=group_by
        ).apply(lambda df: df.sort_values(by='metric_value').iloc[0])
metrics_long_sel_min


# %%
def get_plotly_figure(dataset: str, x='latent_dim'):
    fig = px.scatter(metrics_long_sel_min.loc[dataset],
                     x=x,
                     y='metric_value',
                     color="model",
                     facet_row="metric_name",
                     facet_col="subset_w_N",
                     hover_data=hover_data,
                     title=f'Performance on {dataset.replace("_", " ")} data',
                     labels=labels_dict,
                     category_orders=category_orders,
                     width=1600,
                     height=700,
                     template='none',
                     )
    fig.update_xaxes(dict(
        tickmode='array',
        tickvals=sorted(metrics_long[x].unique()),
    )
    )
    fig.update_layout(
        font={'size': 18},
        xaxis={'title': {'standoff': 15}},
        yaxis={'title': {'standoff': 15}})
    return fig


dataset = 'test_fake_na'
fig = get_plotly_figure(dataset)
fname = FOLDER / f"hyperpar_{dataset}_results_best.pdf"
files_out[f"hyperpar_{dataset}_results_best.pdf"] = fname
fig.write_image(fname)
logger.info(f"Save to {fname}")
fig.show()

# %%
dataset = 'valid_fake_na'
fig = get_plotly_figure(dataset)
fname = FOLDER / f"hyperpar_{dataset}_results_best.pdf"
files_out[f"hyperpar_{dataset}_results_best.pdf"] = fname
fig.write_image(fname)
logger.info(f"Save to {fname}")
fig.show()

# %% [markdown]
# ## Performance along feature prevalence in training data

# %%
dataset = 'valid_fake_na'
group_by = ['data_split', 'subset', 'metric_name', 'model', 'latent_dim']
METRIC = 'MAE'
selected = (metrics_long
            .reset_index()
            .groupby(by=group_by)
            .apply(lambda df: df.sort_values(by='metric_value').iloc[0])
            .loc[dataset])
fname = FOLDER / 'best_models_metrics_per_latent.csv'
files_out['best_models_metrics_per_latent.csv'] = fname
selected.to_csv(fname)
selected.sample(5)

# %%
model_with_latent = list(selected['model'].unique())
model_with_latent

# %% [markdown]
# ### For best latent dimension (on average)

# %% [markdown]
# select minimum value of latent dim over trained models on average
#  1. select for each latent the best model configuration (by DL model)
#  2. Choose the on average best model

# %%
min_latent = (selected.loc['NA interpolated']
                      .loc[METRIC]
                      .loc[model_with_latent]
                      .groupby(level='latent_dim')
                      .agg({'metric_value': 'mean'})
                      .sort_values('metric_value')
              )
min_latent

# %%
min_latent = min_latent.index[0]
print("Minimum latent value for average of models:", min_latent)

# %%
selected = (selected
            .loc['NA interpolated']
            .loc['MAE']
            .loc[model_with_latent]
            .loc[pd.IndexSlice[:, min_latent], :]
            )
selected

# %% [markdown]
# load predictions (could be made better)

# %%
dataset = 'test_fake_na'  # load test split predictions
selected['pred_to_load'] = (
    selected['out_preds']
    + ('/pred_val_' if 'val' in dataset else '/pred_test_')  # not good...
    # + selected['hidden_layers'].apply(lambda s: '_hl_' + '_'.join(str(x)
    #                                   for x in s) + '_' if s is not None else '_')
    + selected.model
    + '.csv'
)
selected['pred_to_load'].to_list()

# %%
selected

# %%
mapper = {k: f'{k} - ' + "HL: {}".format(
    str(selected.loc[k, 'hidden_layers'].to_list()[0]))
    for k in selected.model
}
mapper

# %%
_order = ['observed'] + [m for m in category_orders['model'] if m in selected['model']]
_order

# %%
pred_split = compare_predictions.load_predictions(
    selected['pred_to_load'].to_list())[[*_order]]
pred_split = pred_split.rename(mapper, axis=1)
category_orders['model'] = list(pred_split.columns[1:])
pred_split

# %%
data = datasplits.DataSplits.from_folder(FOLDER / 'data', file_format='pkl')

N_SAMPLES = int(data.train_X.index.levels[0].nunique())
# selection criteria # maybe to be set externally (depends on data selection)
FREQ_MIN = int(N_SAMPLES * 0.25)

logger.info(
    f"N Samples: {N_SAMPLES:,d} - set minumum: {FREQ_MIN:,d} for plotting.")

# %%
freq_feat = sampling.frequency_by_index(data.train_X, 0)
freq_feat.name = 'freq'
# freq_feat = vaep.io.datasplits.load_freq(data_folder) # could be loaded from datafolder
freq_feat.head()  # training data

# %%
errors = vaep.pandas.calc_errors_per_feat(
    pred=pred_split, freq_feat=freq_feat, target_col='observed')
errors

# %%
files_out[f'n_obs_error_counts_{dataset}.pdf'] = (FOLDER /
                                                  f'n_obs_error_counts_{dataset}.pdf')
ax = (errors['n_obs']
      .value_counts()
      .sort_index()
      .plot(style='.',
       xlabel='number of samples',
       ylabel='observations')
)
vaep.savefig(ax.get_figure(),  files_out[f'n_obs_error_counts_{dataset}.pdf'])

# %%
ax = errors.plot.scatter('freq', 'n_obs')

# %%
n_obs_error_is_based_on = errors['n_obs']
errors = errors.drop('n_obs', axis=1)

# %%
M_feat = len(errors)
window_size = int(M_feat / 50)


# %%
errors_smoothed = errors.copy()
# errors_smoothed[errors.columns[:-1]] = errors[errors.columns[:-1]].rolling(window=window_size, min_periods=1).mean()
errors_smoothed[category_orders['model']] = errors[category_orders['model']].rolling(
    window=window_size, min_periods=1).mean()
errors_smoothed

# %%
errors_smoothed

# %%
mask = errors_smoothed[freq_feat.name] >= FREQ_MIN
ax = errors_smoothed.loc[mask].rename_axis('', axis=1).plot(x=freq_feat.name,
                                                            xlabel='freq/feature prevalence (across samples)',
                                                            ylabel=f'rolling average error ({METRIC})',
                                                            xlim=(
                                                                FREQ_MIN, errors_smoothed[freq_feat.name].max()),
                                                            # title=f'Rolling average error by feature frequency {msg_annotation}'
                                                            )

msg_annotation = f"(Latend dim: {min_latent}, No. of feat: {M_feat}, window_size: {window_size})"
print(msg_annotation)

files_out[f'best_models_ld_{min_latent}_rolling_errors_by_freq'] = (FOLDER /
                                                                    f'best_models_ld_{min_latent}_rolling_errors_by_freq')
vaep.savefig(
    ax.get_figure(),
    name=files_out[f'best_models_ld_{min_latent}_rolling_errors_by_freq'])

# %%
errors_smoothed_long = errors_smoothed.drop('freq', axis=1).stack().to_frame(
    'rolling error average').reset_index(-1).join(freq_feat)
errors_smoothed_long

# %% [markdown]
# Save html versin of curve with annotation of errors

# %%
fig = px_vaep.line(errors_smoothed_long.loc[errors_smoothed_long[freq_feat.name] >= FREQ_MIN].join(n_obs_error_is_based_on).sort_values(by='freq'),
                   x=freq_feat.name,
                   color='model',
                   y='rolling error average',
                   hover_data=['n_obs'],
                   # title=f'Rolling average error by feature frequency {msg_annotation}',
                   labels=labels_dict,
                   category_orders=category_orders,
                   )
fig = px_vaep.apply_default_layout(fig)
fig.update_layout(legend_title_text='')  # remove legend title
files_out[f'best_models_ld_{min_latent}_errors_by_freq_plotly.html'] = (FOLDER /
                                                                        f'best_models_ld_{min_latent}_errors_by_freq_plotly.html')
fig.write_html(
    files_out[f'best_models_ld_{min_latent}_errors_by_freq_plotly.html'])
fig

# %% [markdown]
# #### Average error by feature frequency.
# Group all features with same frequency and calculate average


# %%
errors_smoothed = errors.copy()

ax = errors_smoothed.loc[errors_smoothed['freq'] >= FREQ_MIN].groupby(by='freq'
                                                                      ).mean(
).sort_index(
).rolling(window=3, min_periods=1
          ).mean(
).rename_axis('', axis=1
              ).plot(
    xlabel='freq/ feature prevalence (across samples)',
    ylabel='rolling error average',
    # title='mean error for features averaged for each frequency'
    xlim=(FREQ_MIN, freq_feat.max())
)
files_out[f'best_models_ld_{min_latent}_errors_by_freq_averaged'] = (FOLDER /
                                                                     f'best_models_ld_{min_latent}_errors_by_freq_averaged')
vaep.savefig(
    ax.get_figure(),
    files_out[f'best_models_ld_{min_latent}_errors_by_freq_averaged'])

# %% [markdown]
# ### For best models per model class
#
# - select on validation data, report on prediction on test data

# %%
group_by = ['data_split', 'subset', 'metric_name', 'model']

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['Median', 'interpolated', 'CF', 'DAE', 'VAE']}
order_models = order_categories['model']

# %%
dataset = 'valid_fake_na'  # select on validation split

selected = metrics_long.reset_index(
).groupby(by=group_by
          ).apply(lambda df: df.sort_values(by='metric_value').iloc[0]).loc[dataset]
selected = selected.loc[pd.IndexSlice['NA interpolated',
                                      'MAE']].loc[order_models]
selected.to_csv(FOLDER / 'best_models_metrics.csv')
selected

# %%
order_models = [m for m in order_models if m in selected['model']]
selected = selected.loc[order_models]
selected

# %%
dataset = 'test_fake_na'  # load test split predictions
selected['pred_to_load'] = (
    selected['out_preds']
    + ('/pred_val_' if 'val' in dataset else '/pred_test_')  # not good..."
    # + selected['hidden_layers'].apply(lambda s: '_hl_' + '_'.join(str(x)
    #                                   for x in s) + '_' if s is not None else '_')
    + selected.model
    + '.csv'
)
selected['pred_to_load'].to_list()

# %%
sel_pred_to_load = []

for fname in selected['pred_to_load']:
    fname = pathlib.Path(fname)
    if fname.exists():
        sel_pred_to_load.append(fname.as_posix())
    else:
        logger.warning(f"Missing prediction file: {fname}")
sel_pred_to_load

# %%
mapper = {k: f'{k} - LD: {selected.loc[k, "latent_dim"]} - HL: {selected.loc[k, "hidden_layers"]} '
          for k in selected.model
          }
mapper

# %%
pred_split = compare_predictions.load_predictions(
    sel_pred_to_load)[['observed', *order_models]]
pred_split = pred_split.rename(mapper, axis=1)
order_models = list(pred_split.columns[1:])
pred_split

# %%
feat_count = pred_split.groupby(
    by=pred_split.index.names[-1])[pred_split.columns[0]].count()
ax = feat_count.hist(legend=False)
ax.set_xlabel('feat used for comparison (in split)')
ax.set_ylabel('observations')

# %%
# loaded above
freq_feat

# %%
errors = vaep.pandas.calc_errors_per_feat(
    pred=pred_split, freq_feat=freq_feat, target_col='observed')
idx_name = errors.index.name
errors

# %%
files_out[f'best_models_errors_counts_obs_{dataset}.pdf'] = (FOLDER /
                                                             f'n_obs_error_counts_{dataset}.pdf')
ax = errors['n_obs'].value_counts().sort_index().plot(style='.')
vaep.savefig(ax.get_figure(),
             files_out[f'best_models_errors_counts_obs_{dataset}.pdf'])

# %%
n_obs_error_is_based_on = errors['n_obs']
errors = errors.drop('n_obs', axis=1)

# %%
# shoudl be the same
M_feat = len(errors)
window_size = int(M_feat / 50)
print(
    f"Features in split: {M_feat}, set window size for smoothing: {window_size}")
msg_annotation = f"(No. of feat: {M_feat}, window_size: {window_size})"

# %%
errors_smoothed = errors.copy()
errors_smoothed[order_models] = errors[order_models].rolling(
    window=window_size, min_periods=1).mean()
mask = errors_smoothed[freq_feat.name] >= FREQ_MIN
ax = errors_smoothed.loc[mask].rename_axis('', axis=1).plot(x=freq_feat.name,
                                                            ylabel='rolling error average',
                                                            xlabel='freq/feature prevalence (across samples)',
                                                            xlim=(
                                                                FREQ_MIN, freq_feat.max()),
                                                            # title=f'Rolling average error by feature frequency {msg_annotation}'
                                                            )

vaep.savefig(
    ax.get_figure(),
    folder=FOLDER,
    name=f'best_models_rolling_errors_{dataset}')

# %%
errors_smoothed_long = errors_smoothed.drop('freq', axis=1).stack().to_frame(
    'rolling error average').reset_index(-1).join(freq_feat).join(feat_count).reset_index()
errors_smoothed_long

# %% [markdown]
# Save html versin of curve with annotation of errors

# %%
fig = px_vaep.line((errors_smoothed_long.loc[errors_smoothed_long[freq_feat.name] >= FREQ_MIN]
                                        .join(n_obs_error_is_based_on)
                                        .sort_values(by='freq')),
                   x=freq_feat.name,
                   color='model',
                   y='rolling error average',
                   title=f'Rolling average error by feature frequency {msg_annotation}',
                   labels=labels_dict,
                   hover_data=[feat_count.name, idx_name, 'n_obs'],
                   category_orders={'model': order_models})
fig = px_vaep.apply_default_layout(fig)
fig.update_layout(legend_title_text='')  # remove legend title
files_out[f'best_models_errors_{dataset}_by_freq_plotly.html'] = (FOLDER /
                                                                  f'best_models_errors_{dataset}_by_freq_plotly.html')
# fig.write_image(FOLDER / f'best_models_errors_{dataset}_by_freq_plotly.pdf')
fig.write_html(files_out[f'best_models_errors_{dataset}_by_freq_plotly.html'])
fig

# %% [markdown]
# ## Correlation plots

# %%
pred_split

# %% [markdown]
# ### by feature across samples

# %%
corr_per_feat = pred_split.groupby(idx_name).aggregate(
    lambda df: df.corr().loc['observed'])[order_models]
corr_per_feat = corr_per_feat.join(pred_split.groupby(idx_name)[
                                   'observed'].count().rename('n_obs'))
too_few_obs = corr_per_feat['n_obs'] < 3
corr_per_feat.describe()

# %%
corr_per_feat.loc[~too_few_obs].describe()

# %%
corr_per_feat.loc[too_few_obs].dropna(thresh=3, axis=0)

# %%
figsize = 8, 8  # None
fig, ax = plt.subplots(figsize=figsize)

kwargs = dict(rot=45,
              # title='Corr. betw. simulated NA and model pred. per feat',
              ylabel=f'correlation per feature ({idx_name})')
ax = corr_per_feat.loc[~too_few_obs].drop(
    'n_obs', axis=1).plot.box(**kwargs, ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
files_out[f'pred_corr_per_feat_{dataset}'] = (FOLDER /
                                              f'pred_corr_per_feat_{dataset}')
vaep.savefig(ax.get_figure(), name=files_out[f'pred_corr_per_feat_{dataset}'])

# %%
files_out[f'pred_corr_per_feat_{dataset}.xlsx'] = (FOLDER /
                                                   f'pred_corr_per_feat_{dataset}.xlsx')
with pd.ExcelWriter(files_out[f'pred_corr_per_feat_{dataset}.xlsx']) as writer:
    corr_per_feat.loc[~too_few_obs].describe().to_excel(
        writer, sheet_name='summary')  # excluded -1 and 1 version
    # complete information
    corr_per_feat.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# ### within sample

# %%
corr_per_sample = pred_split.groupby('Sample ID').aggregate(
    lambda df: df.corr().loc['observed'])[order_models]
corr_per_sample = corr_per_sample.join(pred_split.groupby('Sample ID')[
                                       'observed'].count().rename('n_obs'))
corr_per_sample.describe()

# %%
fig, ax = plt.subplots(figsize=figsize)

kwargs = dict(ylim=(0.7, 1), rot=45,
              # title='Corr. betw. simulated NA and model pred. per sample',
              ylabel='correlation per sample')
ax = corr_per_sample.drop('n_obs', axis=1).plot.box(**kwargs, ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
files_out[f'pred_corr_per_sample_{dataset}'] = (FOLDER /
                                                f'pred_corr_per_sample_{dataset}')
vaep.savefig(ax.get_figure(),
             name=files_out[f'pred_corr_per_sample_{dataset}'])

# %%
files_out[f'pred_corr_per_sample_{dataset}.xlsx'] = (FOLDER /
                                                     f'pred_corr_per_sample_{dataset}.xlsx')
with pd.ExcelWriter(files_out[f'pred_corr_per_sample_{dataset}.xlsx']) as writer:
    corr_per_sample.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample.to_excel(writer, sheet_name='correlations')

# %% [markdown]
# # Files written to disk

# %%
files_out
