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
metrics_json:str = "path/to/all_metrics.json" # file path to metrics json
configs_json:str = "path/to/all_configs.json" # file path to configs json ("meta data")

# %%
try:
    assert pathlib.Path(metrics_json).exists()
    assert pathlib.Path(configs_json).exists()
except AssertionError:
    metrics_json = snakemake.input.metrics
    configs_json = snakemake.input.config
    print(f"{metrics_json = }", f"{configs_json = }", sep="\n")  

# %% [markdown]
# ## Load metrics

# %%
path_metrics_json = pathlib.Path(metrics_json)
path_configs_json = pathlib.Path(configs_json)
FOLDER = path_metrics_json.parent

metrics_dict = vaep.io.load_json(path_metrics_json)
configs_dict = vaep.io.load_json(path_configs_json)

# %% [markdown]
# Random sample metric schema (all should be the same)

# %%
key_sampled = vaep.utils.sample_iterable(metrics_dict, 1)[0]
key_map = vaep.pandas.key_map(metrics_dict[key_sampled])
key_map

# %% [markdown]
# Metrics a `pandas.DataFrame`:

# %% tags=[]
metrics_dict_multikey = {}
for k, run_metrics in metrics_dict.items():
    metrics_dict_multikey[k] = {eval(k): v for k, v in run_metrics.items()} #vaep.pandas.flatten_dict_of_dicts(run_metrics)

# metrics_dicts = {
#     'AEs': {k:v for k,v in metrics_dict_multikey.items() if 'collab' not in k},
#     'collab': {k:v for k,v in metrics_dict_multikey.items() if 'collab' in k}
# }

metrics = pd.DataFrame.from_dict(metrics_dict_multikey, orient='index')
metrics.columns.names = ['subset','data_split', 'model', 'metric_name']
metrics.index.name = 'id'
metrics = metrics.dropna(axis=1,how='all')
metrics = metrics.stack('model')
metrics = metrics.drop_duplicates()
metrics

# %%
metrics.sort_values(by=('NA interpolated', 'valid_fake_na', 'MAE'))

# %%
# sort_by = 'MAE'
metric_columns = ['MSE', 'MAE']
model_keys = ['collab', 'dae', 'vae']
subset = metrics.columns.levels[0][0]
print(f"{subset = }")

# %% [markdown]
# ## Metadata

# %% [markdown]
# Experiment metadata from configs

# %%
meta = pd.read_json(path_configs_json).T
meta['hidden_layers'] = meta.loc[meta['hidden_layers'].notna() ,'hidden_layers'].apply(tuple) # make list a tuple
meta['n_hidden_layers'] = meta.hidden_layers.loc[meta['hidden_layers'].notna()].apply(len).fillna(0)

mask_collab = meta.index.str.contains('collab')
meta.loc[mask_collab, 'batch_size'] = meta.loc[mask_collab, 'batch_size_collab']
meta.loc[mask_collab, 'hidden_layers'] = None

meta

# %% [markdown]
# Batch size for collab models depends on a factor (as the data in long format has roughly  N samples * M features entries).

# %% [markdown] tags=[]
# ### Plot Top 10 for simulated NA validation data
# - options see [2Dline plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D)

# %%
ax = metrics[subset]["valid_fake_na"].sort_values(
    'MSE').iloc[:10, :-1].plot(rot=70, 
                          x_compat=True, 
                          xticks=list(range(10)),
                          marker='o',
                          linestyle='',
                          title='Top 10 results for hyperparameters',
                         )
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# %%
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='top_10_models_validation_fake_na', folder=FOLDER)

# %% [markdown]
# ## Colorcoded metrics
#
# - can be one of the [matplotlib color maps](https://matplotlib.org/stable/tutorials/colors/colormaps.html), which also have reversed version indicated by `*_r`
#
# ``` python
# ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
# ```

# %%
cmap='cividis_r'

# %%
metrics_styled = metrics.unstack('model')

metrics_styled = (
    metrics_styled.set_index(
        pd.MultiIndex.from_frame(
            meta.loc[metrics_styled.index, ['latent_dim', 'hidden_layers', 'batch_size']]
        ))
    .sort_index()
    .stack('model')
    .style.background_gradient(cmap)
)
metrics = metrics_styled.data
metrics_styled

# %%
metrics_styled.to_excel(FOLDER/ 'metrics_styled.xlsx')

# %%
for k in metrics.columns.levels[0][::-1]:
    print("\n"+"*"*10, f"Subset: {k}\n")
    display(metrics[k].style.background_gradient(cmap))

# %% [markdown]
# ### Plot Top 10 for simulated Na validation data

# %%
ax = metrics[subset]["valid_fake_na"].sort_values(
    'MSE').iloc[:10,:-1].plot(rot=45,
                          x_compat=False,
                          xticks=list(range(10)),
                          marker='o',
                          linestyle='',
                          )
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, name='top_10_models_validation_fake_na_v02', folder=FOLDER)

# %% [markdown]
# ## Collection of Performance plots 
#
# - similar to hyperparameter performance plots in Tensorboard

# %%
metrics = metrics.unstack('model').reset_index()
metrics.iloc[:, :7].head(3)


# %% [markdown]
# ### Parallel coordinates
#
# - similar to Tensorboard visualization of a set of hyperparameters

# %%
def plot_parallel_categories(metrics=metrics, model_type='DAE', metric_type='MSE', subset='NA interpolated', split='valid_fake_na'):
    sel_metric = (subset, split , metric_type, model_type)
    metric_sel = metrics.loc[:, [('latent_dim', '', '', ''),
                                ('hidden_layers', '', '', ''),
                                sel_metric]].dropna()
    title = ' '.join(sel_metric)
    metric_sel.columns = [' '.join(x[0].split('_'))
                        for x in metric_sel.columns[:-1]] + [sel_metric[-2]]
    fig = px.parallel_categories(metric_sel, dimensions=metric_sel.columns[:-1],
                color="MSE", color_continuous_scale=px.colors.sequential.Inferno,
                title=title
    )
    
    return fig

fig = plot_parallel_categories(model_type='DAE')
fig.show()

# %%
fig = plot_parallel_categories(metrics, 'VAE')
fig.show()

# %% [markdown]
# ### Plotting without Multi-Index

# %%
metrics = {k: vaep.pandas.create_dict_of_dicts(d) for k, d in metrics_dict_multikey.items()}
metrics = pd.json_normalize([{'index': k, **d} for k,d in metrics.items()], meta='index', sep= ' ')
metrics = metrics.set_index('index')
metrics = meta.join(metrics)
metrics

# %%
labels_dict = {"NA not interpolated valid_collab collab MSE": 'MSE',
               'batch_size_collab': 'bs',
               'n_hidden_layers': "No. of hidden layers",
               'latent_dim': 'hidden layer dimension',
               'subset_w_N': 'subset',
               'n_params': 'no. of parameter',
               "metric_value": 'value',
               'metric_name': 'metric',
               'freq': 'freq/feature prevalence (across samples)'}

# %% [markdown]
# #### Single model metric - collab

# %% [markdown]
# ### Plotting from long format
#
# To use colors meaningfully, the long format of the data is needed.

# %%
metrics_long = pd.DataFrame.from_dict(metrics_dict_multikey, orient='index')
columns_names = ['subset', 'data_split', 'model', 'metric_name']
metrics_long.columns.names = columns_names
metrics_long.index.name = 'id'
metrics_long

# %% [markdown]
# Combine N into single column

# %%
metrics_N = metrics_long.loc[:, pd.IndexSlice[:,:,:, 'N']]
metrics_N = metrics_N.stack(['subset', 'data_split', 'model', 'metric_name']).unstack('metric_name').astype(int)
metrics_N #.unstack(['subset', 'data_split', 'model',])

# %% [markdown]
# join N used to compute metric

# %%
metrics_long=metrics_long.loc[:, pd.IndexSlice[:,:,:, metric_columns]]
metrics_long = metrics_long.stack(metrics_long.columns.names).to_frame('metric_value').reset_index('metric_name').join(metrics_N)
metrics_long

# %% [markdown]
# join metadata for each metric

# %%
metrics_long = metrics_long.reset_index(['subset', 'data_split', 'model']).join(meta)
metrics_long.index.name = 'id'
metrics_long

# %% [markdown]
# Combine number of parameters into one columns (they are mutually exclusive)

# %%
metrics_long['n_params'] = metrics_long['n_params_collab']
for key in ['DAE', 'VAE']:
    mask = metrics_long.model == key
    metrics_long.loc[mask, 'n_params'] = metrics_long.loc[mask, f'n_params_{key.lower()}']
mask = metrics_long.model == 'interpolated'
metrics_long.loc[mask, 'n_params'] = 1 # at least overall (and 1 for the number of replicates?)
mask = metrics_long.model == 'median'
metrics_long.loc[mask, 'n_params'] = metrics_long.loc[mask, 'M'] # number of features to calculate median of

metrics_long[[*columns_names, 'n_params', 'n_params_vae', 'n_params_dae', 'n_params_collab']]

# %%
metrics_long['subset_w_N'] = metrics_long['subset'].str[0:] + ' - N: ' + metrics_long['N'].astype(str)
metrics_long[['subset_w_N', 'subset']]

# %%
metrics_long.to_csv(FOLDER / 'metrics_long_df.csv') # Should all the plots be done without the metrics?

# %%
category_orders = {'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE'],
                   }

# %%
col = "NA interpolated valid_fake_na collab MAE"
# col = ("NA interpolated","valid_fake_na","collab","MSE")
fig = px.scatter(metrics_long.query('model == "collab"'),
                 x="latent_dim",
                 y='metric_value',
                 color="subset", # needs data in long format
                 facet_row="metric_name",
                 facet_col="data_split",
                 title='Performance of collaborative filtering models',
                 labels={**labels_dict, 'data_split': 'data split'},
                 category_orders={'data_split': ['valid_fake_na', 'test_fake_na']},
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
fig.write_image(FOLDER / 'collab_performance_overview.pdf')
fig.show()


# %% [markdown]
# ## Plot hyperparameter results - overview

# %%
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
                     hover_data=['hidden_layers', 'latent_dim'],
                     #             'batch_size', 'batch_size_collab',
                     # 'subset', f'NA not interpolated {dataset} N', f'NA interpolated {dataset} N'],
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
fig.write_image(FOLDER / f"hyperpar_{dataset}_results_by_parameters_all.pdf")
fig

# %%
fig = plot_by_params('', subset='NA interpolated')
fig.write_image(FOLDER / f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf")
fig

# %%
dataset = "test_fake_na"
fig = plot_by_params(dataset, 'NA interpolated')
fig.write_image(FOLDER / f"hyperpar_{dataset}_results_by_parameters_na_interpolated.pdf")
fig

# %% [markdown]
# ### select best run (->minimum loss) for criteria compared:

# %%
group_by = ['data_split','subset', 'latent_dim', 'metric_name', 'model']
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
                     hover_data=['n_hidden_layers', 'hidden_layers',
                                 'batch_size', 'batch_size_collab',
                                 'n_params'
                                ],
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
fig.write_image(FOLDER / f"hyperpar_{dataset}_results_best.pdf")
fig.show()

# %%
dataset = 'valid_fake_na'
fig = get_plotly_figure(dataset)
fig.write_image(FOLDER / f"hyperpar_{dataset}_results_best.pdf")
fig.show()

# %% [markdown]
# ## Plot for best models predictions along completeness

# %%
dataset = 'valid_fake_na'
group_by = ['data_split', 'subset', 'metric_name', 'model', 'latent_dim' ]
METRIC = 'MAE'
selected = metrics_long.reset_index(
    ).groupby(by=group_by
    ).apply(lambda df: df.sort_values(by='metric_value').iloc[0]).loc[dataset]
selected.to_csv(FOLDER / 'best_models_metrics_per_latent.csv')
selected

# %% [markdown]
# ### For best latent dimension (on average)

# %% [markdown]
# select minimum value of latent dim over trained models on average
#  1. select for each latent the best model configuration (by DL model)
#  2. Choose the on average best model

# %%
min_latent = selected.loc['NA interpolated'].loc[METRIC].loc[['DAE', 'VAE', 'collab']].groupby(level='latent_dim').mean().sort_values('metric_value')
min_latent

# %%
min_latent = min_latent.index[0]
print("Minimum latent value for average of models:", min_latent)

# %%
selected = selected.loc['NA interpolated'].loc['MAE'].loc[['collab', 'DAE', 'VAE']].loc[pd.IndexSlice[:, min_latent], :]
selected

# %% [markdown]
# load predictions (could be made better)

# %%
selected['pred_to_load'] = (
    selected['out_preds']
    + ('/pred_val' if 'val' in dataset else '/pred_test_')  # not good...
    + selected['hidden_layers'].apply(lambda s: '_hl_' + '_'.join(str(x) for x in s) + '_' if s is not None else '_')
    + selected.model.str.lower()
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
pred_split = compare_predictions.load_predictions(selected['pred_to_load'].to_list())[['observed', *category_orders['model']]]
pred_split = pred_split.rename(mapper, axis=1)
category_orders['model'] = list(pred_split.columns[1:])
pred_split

# %%
data = datasplits.DataSplits.from_folder(FOLDER / 'data', file_format='pkl')

N_SAMPLES = int(data.train_X.index.levels[0].nunique())
FREQ_MIN = int(N_SAMPLES * 0.25) # selection criteria # maybe to be set externally (depends on data selection)

logger.info(f"N Samples: {N_SAMPLES:,d} - set minumum: {FREQ_MIN:,d} for plotting.")

# %%
freq_feat = sampling.frequency_by_index(data.train_X, 0)
freq_feat.name = 'freq'
# freq_feat = vaep.io.datasplits.load_freq(data_folder) # could 
freq_feat.head() # training data

# %%
errors_val = vaep.pandas.calc_errors_per_feat(pred=pred_split, freq_feat=freq_feat, target_col='observed')
errors_val

# %%
M_feat = len(errors_val)
window_size = int(M_feat / 50)
print(f"Features in split: {M_feat}, set window size for smoothing: {window_size}")
msg_annotation = f"(Latend dim: {min_latent}, No. of feat: {M_feat}, window_size: {window_size})"

# %%
errors_val_smoothed = errors_val.copy()
# errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]].rolling(window=window_size, min_periods=1).mean()
errors_val_smoothed[category_orders['model']] = errors_val[category_orders['model']].rolling(window=window_size, min_periods=1).mean()
errors_val_smoothed

# %%
mask = errors_val_smoothed[freq_feat.name] >= FREQ_MIN
ax = errors_val_smoothed.loc[mask].rename_axis('', axis=1).plot(x=freq_feat.name,
                                        xlabel='freq/feature prevalence (across samples)',
                                        ylabel=f'rolling average error ({METRIC})',
                                        xlim=(FREQ_MIN, errors_val_smoothed[freq_feat.name].max()),
                                        # title=f'Rolling average error by feature frequency {msg_annotation}'
                                       )

vaep.savefig(
    ax.get_figure(),
    folder=FOLDER,
    name=f'best_models_ld_{min_latent}_rolling_errors_by_freq')

# %%
errors_val_smoothed_long = errors_val_smoothed.drop('freq', axis=1).stack().to_frame('rolling error average').reset_index(-1).join(freq_feat)
errors_val_smoothed_long

# %% tags=[]
fig = px_vaep.line(errors_val_smoothed_long.loc[errors_val_smoothed_long[freq_feat.name] >= FREQ_MIN].sort_values(by='freq'),
              x=freq_feat.name,
              color='model',
              y='rolling error average',
              # title=f'Rolling average error by feature frequency {msg_annotation}',
              labels=labels_dict,
              category_orders=category_orders,
              )
fig = px_vaep.apply_default_layout(fig)
fig.update_layout(legend_title_text='') # remove legend title
fig.write_image(FOLDER / f'best_models_ld_{min_latent}_errors_by_freq_plotly.pdf')
fig

# %%
errors_val_smoothed = errors_val.copy()

ax = errors_val_smoothed.loc[errors_val_smoothed['freq'] >= FREQ_MIN].groupby(by='freq'
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

vaep.savefig(
    ax.get_figure(),
    folder=FOLDER,
    name=f'best_models_ld_{min_latent}_errors_by_freq_averaged')

# %% [markdown]
# ### For best models per model class

# %%
# dataset = 'valid_fake_na'
dataset = 'test_fake_na'

group_by = ['data_split', 'subset', 'metric_name', 'model']

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE']}
order_models = order_categories['model']

# %%
selected = metrics_long.reset_index(
    ).groupby(by=group_by
    ).apply(lambda df: df.sort_values(by='metric_value').iloc[0]).loc[dataset]
selected = selected.loc[pd.IndexSlice['NA interpolated', 'MAE']].loc[order_models]
selected.to_csv(FOLDER / 'best_models_metrics.csv')
selected

# %%
selected = selected.loc[['collab', 'DAE', 'VAE']]
selected

# %%
selected['pred_to_load'] = (
    selected['out_preds']
    + ('/pred_val' if 'val' in dataset else '/pred_test')  # not good...
    + selected['hidden_layers'].apply(lambda s: '_hl_' + '_'.join(str(x) for x in s) + '_' if s is not None else '_')
    + selected.model.str.lower()
    + '.csv'
)
selected['pred_to_load'].to_list()

# %%
mapper = {k: f'{k} - LD: {selected.loc[k, "latent_dim"]} - HL: {selected.loc[k, "hidden_layers"]} '
          for k in selected.model
         }
mapper

# %%
pred_split = compare_predictions.load_predictions(selected['pred_to_load'].to_list())[['observed', *order_models]]
pred_split = pred_split.rename(mapper, axis=1)
order_models = list(pred_split.columns[1:])
pred_split

# %%
feat_count=pred_split.groupby(by=pred_split.index.names[-1])[pred_split.columns[0]].count()
ax = feat_count.hist(legend=False)
ax.set_xlabel('feat used for comparison (in split)')
ax.set_ylabel('observations')

# %%
# loaded above
freq_feat

# %%
errors_val = vaep.pandas.calc_errors_per_feat(pred=pred_split, freq_feat=freq_feat, target_col='observed')
idx_name = errors_val.index.name
errors_val

# %%
# shoudl be the same
M_feat = len(errors_val)
window_size = int(M_feat / 50)
print(f"Features in split: {M_feat}, set window size for smoothing: {window_size}")
msg_annotation = f"(No. of feat: {M_feat}, window_size: {window_size})"

# %%
errors_val_smoothed = errors_val.copy()
errors_val_smoothed[order_models] = errors_val[order_models].rolling(window=window_size, min_periods=1).mean()
mask = errors_val_smoothed[freq_feat.name] >= FREQ_MIN
ax = errors_val_smoothed.loc[mask].rename_axis('', axis=1).plot(x=freq_feat.name,
                                        ylabel='rolling error average',
                                        xlabel='freq/feature prevalence (across samples)',
                                        xlim=(FREQ_MIN,freq_feat.max()),
                             # title=f'Rolling average error by feature frequency {msg_annotation}'
                                       )

vaep.savefig(
    ax.get_figure(),
    folder=FOLDER,
    name=f'best_models_rolling_errors_{dataset}')

# %%
errors_val_smoothed_long = errors_val_smoothed.drop('freq', axis=1).stack().to_frame('rolling error average').reset_index(-1).join(freq_feat).join(feat_count).reset_index()
errors_val_smoothed_long

# %%
fig = px_vaep.line(errors_val_smoothed_long.loc[errors_val_smoothed_long[freq_feat.name] >= FREQ_MIN].sort_values(by='freq'),
                   x=freq_feat.name,
                   color='model',
                   y='rolling error average',
                   title=f'Rolling average error by feature frequency {msg_annotation}',
                   labels=labels_dict,
                   hover_data=[feat_count.name, idx_name],
                   category_orders={'model': order_models})
fig = px_vaep.apply_default_layout(fig)
fig.update_layout(legend_title_text='') # remove legend title
fig.write_image(FOLDER / f'best_models_errors_{dataset}_by_freq_plotly.pdf')
fig.write_html(FOLDER / f'best_models_errors_{dataset}_by_freq_plotly.html')
fig

# %% [markdown]
# #### correlation plots

# %%
pred_split

# %%
corr_per_feat = pred_split.groupby(idx_name).aggregate(lambda df: df.corr().loc['observed'])[order_models]

figsize = 8,8 # None
fig, ax = plt.subplots(figsize=figsize)

kwargs = dict(rot=45,
              # title='Corr. betw. simulated NA and model pred. per feat',
              ylabel=f'correlation per feature ({idx_name})')
ax = corr_per_feat.plot.box(**kwargs, ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
vaep.savefig(ax.get_figure(), name=f'pred_corr_per_feat_{dataset}', folder=FOLDER)
with pd.ExcelWriter(FOLDER/f'pred_corr_test_per_feat_{dataset}.xlsx') as writer:
    corr_per_feat.describe().to_excel(writer, sheet_name='summary')
    corr_per_feat.to_excel(writer, sheet_name='correlations')

# %%
corr_per_sample = pred_split.groupby('Sample ID').aggregate(lambda df: df.corr().loc['observed'])[order_models]
corr_per_sample.describe()

# %%
fig, ax = plt.subplots(figsize=figsize)

kwargs = dict(ylim=(0.7, 1), rot=45,
              # title='Corr. betw. simulated NA and model pred. per sample',
              ylabel='correlation per sample')
ax = corr_per_sample.plot.box(**kwargs, ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
vaep.savefig(ax.get_figure(), name=f'pred_corr_per_sample_{dataset}', folder=FOLDER)
with pd.ExcelWriter(FOLDER/f'pred_corr_per_sample_{dataset}.xlsx') as writer:
    corr_per_sample.describe().to_excel(writer, sheet_name='summary')
    corr_per_sample.to_excel(writer, sheet_name='correlations')

# %%
