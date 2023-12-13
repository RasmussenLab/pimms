# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Best model over datasets

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import vaep.plotting
import vaep.nb


pd.options.display.max_columns = 45
pd.options.display.max_rows = 110
pd.options.display.multi_sparse = False

plt.rcParams['figure.figsize'] = [12.0, 6.0]

vaep.plotting.make_large_descriptors()

logger = vaep.logging.setup_nb_logger()

# %% [markdown]
# ## Read input

# %%
metrics_long = []

for fname in snakemake.input:
    print(f"{fname = }")
    fname = Path(fname)
    level = fname.parent.name

    _df = pd.read_csv(fname)
    _df['data level'] = level

    metrics_long.append(_df)
    del _df
metrics_long = pd.concat(metrics_long)
metrics_long['id'] = metrics_long['data level'] + metrics_long['id'].str[3:]
metrics_long = metrics_long.set_index('id')
metrics_long

# %%
# snakemake.params.folder
try:
    models = snakemake.params.models  # snakefile would need to be
except AttributeError:
    models = ['Median', 'interpolated', 'CF', 'DAE', 'VAE']
models

# %%
group_by = ['data_split', 'data level', 'metric_name', 'model']

selected_cols = ['metric_value', 'latent_dim', 'hidden_layers', 'n_params', 'text', 'N', 'M', 'id']

order_categories = {'data level': ['proteinGroups', 'peptides', 'evidence'],
                    'model': [*models]}

# %%
FOLDER = fname.parent.parent
print(f"{FOLDER =}")

# %% [markdown]
# ## Annotation & Dump of metrics

# %%
metrics_long.loc[metrics_long['model'].isin(['interpolated', 'median']), ['latent_dim', 'hidden_layers']] = '-'
metrics_long['hidden_layers'] = metrics_long['hidden_layers'].fillna('-')
metrics_long['text'] = 'LD: ' + metrics_long['latent_dim'].astype(str) + ' HL: ' + metrics_long['hidden_layers']

# save metrics
fname = 'metrics_long'
metrics_long.to_csv(FOLDER / f'{fname}.csv')
metrics_long.to_excel(FOLDER / f'{fname}.xlsx')

metrics_long[['latent_dim', 'hidden_layers', 'model', 'text', ]]

# %% [markdown]
# ## Settings

# %%
_unique = metrics_long["data level"].unique()
order_categories['data level'] = [l for l in order_categories['data level'] if l in _unique]  # ensure predefined order
_unique = metrics_long['model'].unique()
order_categories['model'] = [m for m in order_categories['model'] if m in _unique]  # ensure predefined order

semi_supervised = [m for m in ['CF', 'DAE', 'VAE'] if m in _unique]
reference = [m for m in ['median', 'interpolated'] if m in _unique]

IDX_ORDER = (order_categories['data level'],
             order_categories['model'])

METRIC = 'MAE'

order_categories

# %% [markdown]
# ## Select best models
#
# - based on validation data
# - report results on test data and validation data

# %%
dataset = 'valid_fake_na'  # data_split
top_n = 3
selected = (metrics_long
            .reset_index()
            .groupby(by=group_by)
            .apply(
                lambda df: df.sort_values(by='metric_value').iloc[:top_n]
            )
            )
sel_on_val = selected.loc[
    pd.IndexSlice[dataset, IDX_ORDER[0], 'MAE', IDX_ORDER[1]],
    selected_cols]
fname = FOLDER / f'sel_on_val_{dataset}_top_{top_n}.xlsx'
writer = pd.ExcelWriter(fname)
sel_on_val.to_excel(writer, sheet_name=f'top_{top_n}')
sel_on_val

# %%
# select best model of top N with least parameters
sel_on_val = (sel_on_val
              .groupby(by=group_by)
              .apply(
                  lambda df: df.sort_values(by='n_params').iloc[0]
              )
              ).loc[
    pd.IndexSlice[dataset, IDX_ORDER[0], 'MAE', IDX_ORDER[1]],
    selected_cols]
sel_on_val.to_excel(writer, sheet_name=f'selected')
writer.close()
sel_on_val

# %% [markdown]
# Retrieve test data values

# %%
sel_on_val = sel_on_val.set_index(['latent_dim', 'hidden_layers', 'id'], append=True)
idx = sel_on_val.droplevel(level='data_split').index
sel_on_val = sel_on_val.reset_index(['latent_dim', 'hidden_layers', 'id'])

test_results = (metrics_long
                .query('data_split == "test_fake_na"')
                .reset_index().set_index(idx.names)
                .loc[idx]
                .reset_index(['latent_dim', 'hidden_layers', 'id'])
                .set_index('data_split', append=True)
                )[selected_cols]
test_results

# %% [markdown]
# ### test data results
#
# - selected on validation data

# %%
test_results = test_results.droplevel(['metric_name']).reset_index().set_index(['model', 'data level'])
test_results

# %%
text = test_results.loc[pd.IndexSlice[IDX_ORDER[1], IDX_ORDER[0]], 'text']
text

# %%
_to_plot = test_results['metric_value'].unstack(0).loc[IDX_ORDER]
_to_plot

# %%
fname = 'best_models_1_test_mpl'
_to_plot.to_excel(FOLDER / f'{fname}.xlsx')
_to_plot.columns.name = ''
ax = _to_plot.plot.bar(rot=0,
                       xlabel='',
                       ylabel=f"{METRIC} (log2 intensities)",
                       width=.8)
ax = vaep.plotting.add_height_to_barplot(ax, size=12)
ax = vaep.plotting.add_text_to_barplot(ax, text, size=12)
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, fname, folder=FOLDER)

# %% [markdown]
# ### Validation data results

# %%
fname = 'best_models_1_val_mpl'

_to_plot = sel_on_val.reset_index(level=['data level', 'model']).loc[[('valid_fake_na', METRIC), ]]

_to_plot = _to_plot.set_index(['data level', 'model'])[['metric_value', 'text']]
_to_plot = _to_plot.loc[IDX_ORDER, :]
_to_plot.index.name = ''
# text = test_results['text'].unstack().loc[IDX_ORDER].unstack()
_to_plot = _to_plot['metric_value'].unstack().loc[IDX_ORDER]
_to_plot.to_csv(FOLDER / f'{fname}.csv')
_to_plot.to_excel(FOLDER / f'{fname}.xlsx')
# display(text.to_frame('text'))
_to_plot

# %%
_to_plot.columns.name = ''
ax = _to_plot.plot.bar(rot=0,
                       xlabel='',
                       ylabel=f"{METRIC} (log2 intensities)",
                       width=.8)
ax = vaep.plotting.add_height_to_barplot(ax, size=12)
ax = vaep.plotting.add_text_to_barplot(ax, text, size=12)
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, fname, folder=FOLDER)

# %%
fname = 'best_models_1_val_plotly'
_to_plot = sel_on_val.reset_index(level=['data level', 'model']).loc[[('valid_fake_na', METRIC), ]]
_to_plot = _to_plot.set_index(['data level', 'model'])
_to_plot[['metric_value', 'latent_dim', 'hidden_layers', 'text']] = _to_plot[[
    'metric_value', 'latent_dim', 'hidden_layers', 'text']].fillna('-')

_to_plot = _to_plot.loc[pd.IndexSlice[IDX_ORDER], :]
_to_plot.to_csv(FOLDER / f"{fname}.csv")
_to_plot.to_excel(FOLDER / f"{fname}.xlsx")
_to_plot[['metric_value', 'latent_dim', 'hidden_layers', 'text', 'N', 'n_params']]

# %%
fig = px.bar(_to_plot.reset_index(),
             x='data level',
             y='metric_value',
             hover_data={'N': ':,d', 'n_params': ':,d'},  # format hover data
             color='model',
             barmode="group",
             text='text',
             labels={'metric_value': f"{METRIC} (log2 intensities)", 'data level': ''},
             category_orders=order_categories,
             template='none',
             height=600)
fig.update_layout(legend_title_text='')
fig.write_html(FOLDER / f"{fname}.html")
fig

# %% [markdown]
# ## Order by best model setup over all datasets
#
# - select best average model on validation data

# %%
group_by = ['model', 'latent_dim', 'hidden_layers']
data_split = 'valid_fake_na'

metrics_long_sel = metrics_long.query(f'data_split == "{data_split}"'
                                      f' & metric_name == "{METRIC}"')

best_on_average = metrics_long_sel.reset_index(
).groupby(by=group_by
          )['metric_value'].mean().sort_values().reset_index(level=group_by[1:])
best_on_average

# %%
best_on_average.to_csv(FOLDER / 'average_performance_over_data_levels.csv')
best_on_average = best_on_average.groupby(group_by[0]).apply(
    lambda df: df.sort_values(by='metric_value').iloc[0]).set_index(group_by[1:], append=True)
best_on_average

# %% [markdown]
# ### Test split results

# %%
fname = 'average_performance_over_data_levels_best_test'
data_split = 'test_fake_na'

metrics_long_sel_test = metrics_long.query(f'data_split == "{data_split}"'
                                           f' & metric_name == "{METRIC}"')

to_plot = (metrics_long_sel_test
           .reset_index().set_index(group_by)
           .loc[best_on_average.index]
           .reset_index().set_index(['model', 'data level'])
           .loc[pd.IndexSlice[order_categories['model'], order_categories['data level']], :])


to_plot = to_plot.reset_index()
to_plot['model annotated'] = to_plot['model'] + ' - ' + to_plot['text']
order_model = to_plot['model annotated'].drop_duplicates().to_list()  # model name with annotation

to_plot = to_plot.drop_duplicates(subset=['model', 'data level', 'metric_value'])
to_plot.to_csv(FOLDER / f"{fname}.csv")
to_plot

# %%
figsize = (10, 8)  # None # (10,8)
fig, ax = plt.subplots(figsize=figsize)
to_plot.columns.name = ''
ax = (to_plot
      .set_index(['model annotated', 'data level'])['metric_value']
      .unstack().rename_axis('', axis=1)
      .loc[order_model, order_categories['data level']]
      .plot.bar(
          # xlabel="model with overall best performance for all datasets",
          xlabel='',
          ylabel="MAE (log2 intensity)",
          rot=45,
          width=.8,
          ax=ax,
          # colormap="Paired",
          color=[
              '#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c',
              '#fb9a99',
              '#e31a1c',
              '#fdbf6f',
              '#ff7f00',
              '#cab2d6',
              '#6a3d9a',
              '#ffff99',
              '#b15928']
      )
      )
ax = vaep.plotting.add_height_to_barplot(ax, size=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fig.tight_layout()
vaep.savefig(fig, fname, folder=FOLDER)

# %% [markdown]
# plotly version with additional information

# %%
fig = px.bar(to_plot,
             x='model',
             y='metric_value',
             color='data level',
             hover_data={'N': ':,d', 'n_params': ':,d'},  # format hover data
             barmode="group",
             color_discrete_sequence=px.colors.colorbrewer.Paired,
             # color_discrete_sequence=['#a6cee3', '#1f78b4', '#b2df8a'],
             text='text',
             labels={'metric_value': f"{METRIC} (log2 intensities)"},
             category_orders=order_categories,
             template='none',
             height=600)
fig.update_xaxes(title='')
fig.write_image(FOLDER / f"{fname}_plotly.pdf")
fig.update_layout(legend_title_text='')
fig

# %% [markdown]
# ### Validation data results

# %%
fname = 'average_performance_over_data_levels_best_val'
to_plot = (metrics_long_sel
           .reset_index().set_index(group_by)
           .loc[best_on_average.index].reset_index()
           .set_index(['model', 'data level'])
           .loc[pd.IndexSlice[order_categories['model'], order_categories['data level']], :]
           )

to_plot = to_plot.reset_index()
to_plot['model annotated'] = to_plot['model'] + ' - ' + to_plot['text']
order_model = to_plot['model annotated'].drop_duplicates().to_list()  # model name with annotation

to_plot = to_plot.drop_duplicates(subset=['model', 'data level', 'metric_value'])
to_plot.to_csv(FOLDER / f"{fname}.csv")
to_plot

# %%
figsize = (10, 8)  # None # (10,8)
fig, ax = plt.subplots(figsize=figsize)
to_plot.columns.name = ''
ax = (to_plot
      .set_index(['model annotated', 'data level'])['metric_value']
      .unstack().rename_axis('', axis=1)
      .loc[order_model, order_categories['data level']]
      .plot.bar(
          # xlabel="model with overall best performance for all datasets",
          xlabel='',
          ylabel="MAE (log2 intensity)",
          rot=45,
          width=.8,
          ax=ax,
          # colormap="Paired",
          color=[
              '#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c',
              '#fb9a99',
              '#e31a1c',
              '#fdbf6f',
              '#ff7f00',
              '#cab2d6',
              '#6a3d9a',
              '#ffff99',
              '#b15928']
      )
      )
ax = vaep.plotting.add_height_to_barplot(ax, size=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fig.tight_layout()
vaep.savefig(fig, fname, folder=FOLDER)

# %% [markdown]
# plotly version with additional information

# %%
fig = px.bar(to_plot,
             x='model',
             y='metric_value',
             color='data level',
             hover_data={'N': ':,d', 'n_params': ':,d'},  # format hover data
             barmode="group",
             color_discrete_sequence=px.colors.colorbrewer.Paired,
             # color_discrete_sequence=['#a6cee3', '#1f78b4', '#b2df8a'],
             text='text',
             labels={'metric_value': f"{METRIC} (log2 intensities)"},
             category_orders=order_categories,
             template='none',
             height=600)
fig.update_xaxes(title='')
fig.write_image(FOLDER / f"{fname}_plotly.pdf")
fig.update_layout(legend_title_text='')
fig
