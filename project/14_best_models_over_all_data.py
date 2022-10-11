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
FOLDER = fname.parent.parent
print(f"{FOLDER =}")

# %%
metrics_long.loc[metrics_long['model'].isin(['interpolated', 'median']), ['latent_dim', 'hidden_layers']] = '-'
metrics_long['hidden_layers'] = metrics_long['hidden_layers'].fillna('-')
metrics_long['text'] = 'LD: ' + metrics_long['latent_dim'].astype(str) + ' HL: ' + metrics_long['hidden_layers']

fname = 'metrics_long'
metrics_long.to_csv(FOLDER / f'{fname}.csv')
metrics_long.to_excel(FOLDER / f'{fname}.xlsx')

metrics_long[['latent_dim', 'hidden_layers', 'model', 'text', ]]

# %%
group_by = ['data_split', 'data level', 'subset', 'metric_name', 'model']

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE']}
IDX_ORDER = (['proteinGroups', 'aggPeptides', 'evidence'],
             ['median', 'interpolated', 'collab', 'DAE', 'VAE'])
METRIC = 'MAE'

# %%
dataset = 'valid_fake_na'

selected = metrics_long.reset_index(
    ).groupby(by=group_by
              ).apply(lambda df: df.sort_values(by='metric_value').iloc[0])
selected.loc[
    pd.IndexSlice[dataset, IDX_ORDER[0], 'NA interpolated', 'MAE', IDX_ORDER[1]],
    ['metric_value', 'latent_dim', 'hidden_layers', 'n_params', 'text', 'N', 'M', 'id']]

# %%
fname = 'best_models_1_mpl'
_to_plot = selected.droplevel(['data level', 'model']).loc[[('valid_fake_na', 'NA interpolated', METRIC), ]]

_to_plot = _to_plot.set_index(['data level', 'model'])[['metric_value', 'text']]
_to_plot = _to_plot.loc[IDX_ORDER,:]
_to_plot.index.name = ''
text = _to_plot['text'].unstack().loc[IDX_ORDER].unstack()
_to_plot = _to_plot['metric_value'].unstack().loc[IDX_ORDER]
_to_plot.to_csv(FOLDER / f'{fname}.csv')
_to_plot.to_excel(FOLDER / f'{fname}.xlsx')
display(text.to_frame('text'))
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
fname = 'best_models_1_plotly'
_to_plot = selected.droplevel(['data level', 'model']).loc[[('valid_fake_na', 'NA interpolated', METRIC), ]]
_to_plot = _to_plot.set_index(['data level', 'model'])[
                              ['metric_value', 'latent_dim', 'hidden_layers', 'text']].fillna('-')

_to_plot = _to_plot.loc[pd.IndexSlice[IDX_ORDER], :]
_to_plot.to_csv(FOLDER / f"{fname}.csv")
_to_plot.to_excel(FOLDER / f"{fname}.xlsx")
_to_plot

# %%
fig = px.bar(_to_plot.reset_index(),
             x='data level',
             y='metric_value',
             color='model',
             barmode="group",
             text='text',
             labels={'metric_value': f"{METRIC} (log2 intensities)", 'data level': ''},
             category_orders=order_categories,
             template='none',
             height=600)
fig.update_layout(legend_title_text='')
fig.write_image(FOLDER / f"{fname}.pdf")
fig

# %% [markdown]
# ## Order by best model setup over all datasets

# %%
group_by = ['model', 'latent_dim', 'hidden_layers']

data_split = 'valid_fake_na'
subset = 'NA interpolated'

metrics_long_sel = metrics_long.query(f'data_split == "{data_split}"'
                                      f' & subset == "{subset}"'
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

# %%
fname = 'average_performance_over_data_levels_best'
to_plot = metrics_long_sel.reset_index().set_index(group_by).loc[best_on_average.index].reset_index().set_index(['model', 'data level']).loc[
    pd.IndexSlice[order_categories['model'], order_categories['data level']], :]
to_plot

# %%
to_plot = to_plot.reset_index()
to_plot['model annotated'] = to_plot['model'] + ' - ' + to_plot['text']
order_model = to_plot['model annotated'].drop_duplicates().to_list()

to_plot = to_plot.drop_duplicates(subset=['model', 'data level', 'metric_value'])
to_plot.to_csv(FOLDER /f"{fname}.csv")
to_plot

# %%
figsize= (10,8) # None # (10,8)
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
          color = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
      )
      )
ax = vaep.plotting.add_height_to_barplot(ax, size=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
fig.tight_layout()
vaep.savefig(fig, fname, folder=FOLDER)

# %%
# ToDo
# plot with annotation
fig = px.bar(to_plot,
             x='model',
             y='metric_value',
             color='data level',
             barmode="group",
             color_discrete_sequence=px.colors.colorbrewer.Paired,
             # color_discrete_sequence=['#a6cee3', '#1f78b4', '#b2df8a'],
             text='text',
             labels={'metric_value': f"{METRIC} (log2 intensities)", 'model': '',},
             category_orders=order_categories,
             template='none',
             height=600)
fig.write_image(FOLDER / f"{fname}_plotly.pdf")
fig.update_layout(legend_title_text='')
fig
