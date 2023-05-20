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
split = 'valid_fake_na'
selected = metrics.loc[pd.IndexSlice[
    split,
    :, :]].stack()
selected

# %%
to_plot = selected.loc[pd.IndexSlice[:, 'MAE', :]]
to_plot = to_plot.stack().unstack(
    REPITITION_NAME).T.describe().loc[['mean', 'std']].T.unstack(0)
to_plot = to_plot.loc[IDX[0], pd.IndexSlice[:, IDX[1]]]
to_plot

# %%
logger.setLevel(20)  # reset debug
ax = to_plot['mean'].plot.bar(rot=0, width=.8, yerr=to_plot['std'])
ax.set_xlabel('')

# %%
to_dump = to_plot.swaplevel(1, 0, axis=1).sort_index(axis=1)
fname = FOLDER / "model_performance_repeated_runs_avg.csv"
to_dump.to_csv(fname)
to_dump.to_excel(fname.with_suffix(".xlsx"))

# %%
split = 'valid_fake_na'
selected = metrics.loc[pd.IndexSlice[
    split,
    :, 'MAE']].stack(1)
# selected.index.names = ('x', 'split', 'model', 'metric', REPITITION_NAME)
selected.stack().to_frame('MAE').reset_index()

# %%
ax = sns.barplot(x='data level',
                 y='MAE',
                 hue='model',
                 order=IDX[0],
                 ci=95,
                 errwidth=1.5,
                 data=selected.stack().to_frame('MAE').reset_index())
ax.set_xlabel('')
fig = ax.get_figure()

# %%
vaep.savefig(fig, FOLDER / "model_performance_repeated_runs.pdf")

# %%
