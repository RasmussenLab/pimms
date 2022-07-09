# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Best model over datasets

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import vaep.plotting

# import vaep.plotting.plotly

import vaep.nb
plt.rcParams['figure.figsize'] = [16.0, 7.0]

vaep.plotting.make_large_descriptors()

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
metrics_long['hidden_layers'] = metrics_long['hidden_layers'].fillna('-')
metrics_long['text'] = 'LD: ' + metrics_long['latent_dim'].astype(str) + ' HL: ' + metrics_long['hidden_layers']
metrics_long.loc[metrics_long['model'].isin(['interpolated', 'median']), 'text'] = '-'

metrics_long[['latent_dim', 'hidden_layers', 'model', 'text', ]]

# %%
group_by = ['data level', 'data_split', 'subset', 'metric_name', 'model']

order_categories = {'data level': ['proteinGroups', 'aggPeptides', 'evidence'],
                    'model': ['median', 'interpolated', 'collab', 'DAE', 'VAE']}
IDX_ORDER = (['proteinGroups', 'aggPeptides', 'evidence'],
             ['median', 'interpolated', 'collab', 'DAE', 'VAE'])
METRIC = 'MAE'

# %%
# dataset = 'valid_fake_na'

selected = metrics_long.reset_index(
    ).groupby(by=group_by
              ).apply(lambda df: df.sort_values(by='metric_value').iloc[0])
selected

# %%
_to_plot = selected.droplevel(['data level', 'model']).loc[[('valid_fake_na', 'NA interpolated', METRIC), ]]
_to_plot = _to_plot.set_index(['data level', 'model'])['metric_value'].unstack()
_to_plot = _to_plot.loc[IDX_ORDER]
_to_plot.index.name = ''
_to_plot

# %%
ax = _to_plot.plot.bar(rot=90, ylabel=METRIC)

# %%
_to_plot = selected.droplevel(['data level', 'model']).loc[[('valid_fake_na', 'NA interpolated', METRIC), ]]
_to_plot = _to_plot.set_index(['data level', 'model'])[
                              ['metric_value', 'latent_dim', 'hidden_layers', 'text']].fillna('-')

# _to_plot['text'] = 'LD: ' + _to_plot['latent_dim'].astype(str) + ' HL: ' + _to_plot['hidden_layers']
# _to_plot.loc[pd.IndexSlice[:, ['interpolated', 'median']], 'text'] = ''

_to_plot = _to_plot.loc[pd.IndexSlice[IDX_ORDER], :]
_to_plot

# %%
px.bar(_to_plot.reset_index(),
       x='data level',
       y='metric_value',
       color='model',
       barmode="group",
       text='text',
       category_orders=order_categories,
       height=600)

# %% [markdown]
# ## Order by best model setup over all datasets

# %%
group_by = ['model', 'latent_dim', 'hidden_layers']

best = metrics_long.query('data_split == "valid_fake_na"'
                          ' & subset == "NA interpolated"'
                          f' & metric_name == "{METRIC}"').reset_index(
    ).groupby(by=group_by
              )['metric_value'].mean()
best

# %%
best.loc['collab']

# %%
best = best.reset_index(level=group_by[1:]).groupby(group_by[0]).min().set_index(group_by[1:], append=True)
best

# %%
to_plot = metrics_long.query('data_split == "valid_fake_na"'
                             ' & subset == "NA interpolated"'
                             f' & metric_name == "{METRIC}"').reset_index().set_index(group_by).loc[best.index].reset_index().set_index(['model', 'data level']).loc[
    pd.IndexSlice[order_categories['model'], order_categories['data level']], :]
to_plot

# %%
to_plot = to_plot.reset_index()
to_plot['model annotated'] = to_plot['model'] + ' - ' + to_plot['text']
order_model = to_plot['model annotated'].drop_duplicates().to_list()
# to_plot = to_plot.set_index(['model annotated', 'data level'])
# to_plot = to_plot['metric_value'].unstack().loc[order_model, order_categories['data level']]
to_plot

# %%
ax = (to_plot
      .set_index(['model annotated', 'data level'])['metric_value']
      .unstack()
      .loc[order_model, order_categories['data level']]
      .plot.bar(
          xlabel="model with overall best performance for all datasets",
          rot=45)
      )

# %%
# ToDo
# plot with annotation
fig = px.bar(to_plot,
             x='model',
             y='metric_value',
             color='data level',
             barmode="group",
             text='text',
             category_orders=order_categories,
             height=600)

# %%
fig.write_image(FOLDER / 'best_models_over_all_data.pdf')
