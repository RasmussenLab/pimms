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

# %%
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import vaep.pandas
import vaep.nb

import logging
import vaep.plotting
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'), level=10)

plt.rcParams['figure.figsize'] = [4.0, 2.0]
vaep.plotting.make_large_descriptors(5)

# %%
IDX = [['proteinGroups', 'peptides', 'evidence'],
       ['CF', 'DAE', 'VAE']]

REPITITION_NAME = snakemake.params.repitition_name  # 'dataset', 'repeat'

metrics_fname = Path(snakemake.input.metrics)
metrics_fname

# %%
FOLDER = metrics_fname.parent
FOLDER

# %%
metrics = pd.read_pickle(metrics_fname)
metrics

# %%
fname = FOLDER / "model_performance_repeated_runs.xlsx"
writer = pd.ExcelWriter(fname)

# %%
split = 'test_fake_na'
selected = metrics.loc[pd.IndexSlice[
    split,
    :, :]].stack()
selected

# %%
min_max_MAE = (selected
               .loc[pd.IndexSlice[:, 'MAE', :]]
               .groupby('model')
               .agg(['min', 'max']))
min_max_MAE.to_excel(writer, sheet_name='min_max_MAE')
min_max_MAE

# %%
to_plot = selected.loc[pd.IndexSlice[:, 'MAE', :]]
to_plot = to_plot.stack().unstack(
    REPITITION_NAME).T.describe().loc[['mean', 'std']].T.unstack(0)
to_plot = to_plot.loc[IDX[0], pd.IndexSlice[:, IDX[1]]]
to_plot

# %%
logger.setLevel(20)  # reset debug
ax = to_plot['mean'].plot.bar(rot=0,
                              width=.8,
                              color=vaep.plotting.defaults.color_model_mapping,
                              yerr=to_plot['std'])
ax.set_xlabel('')

# %%
to_dump = to_plot.swaplevel(1, 0, axis=1).sort_index(axis=1)
to_dump.to_excel(writer, sheet_name='avg')
fname = FOLDER / "model_performance_repeated_runs_avg.csv"
to_dump.to_csv(fname)


# %%
selected = metrics.loc[pd.IndexSlice[
    split,
    :, 'MAE']].stack(1)
view_long = (selected.stack()
             .to_frame('MAE')
             .reset_index())
view_long

# %%
ax = sns.barplot(x='data level',
                 y='MAE',
                 hue='model',
                 order=IDX[0],
                 palette=vaep.plotting.defaults.color_model_mapping,
                 ci=95,
                 errwidth=1.5,
                 data=view_long)
ax.set_xlabel('')
fig = ax.get_figure()

# %%
vaep.savefig(fig, FOLDER / "model_performance_repeated_runs.pdf")

# %%
