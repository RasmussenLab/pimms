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

# %%
import json
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import vaep.pandas
import vaep.nb
import vaep.models

import logging
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'), level=10)

sns.set_theme()

plt.rcParams['figure.figsize'] = [16.0, 7.0]

# %%
IDX =[['proteinGroups', 'aggPeptides', 'evidence'],
      ['median', 'interpolated', 'collab', 'DAE', 'VAE']]

REPITITION_NAME = 'repeat'


# %%
def select_content(s:str):
    s = s.split('metrics_')[1]
    assert isinstance(s, str), f"More than one split: {s}"
    model, repeat = s.split('_')
    return model, int(repeat)
    
test_cases = ['model_metrics_DAE_0',
              'model_metrics_VAE_3',
              'model_metrics_collab_2']
 
for test_case in test_cases:
    print(f"{test_case} = {select_content(test_case)}")

def key_from_fname(fname:Path):
    model, repeat = select_content(fname.stem)
    key = (fname.parents[1].name, repeat)
    return key

# %%
all_metrics = vaep.models.collect_metrics(snakemake.input.metrics, key_from_fname)
metrics = pd.DataFrame(all_metrics).T
metrics.index.names = ('data level', REPITITION_NAME)
metrics

# %%
FOLDER = Path(snakemake.input.metrics[0]).parents[2]
FOLDER

# %%
metrics = metrics.T.sort_index().loc[pd.IndexSlice[['NA interpolated', 'NA not interpolated'],
                                         ['valid_fake_na', 'test_fake_na'],
                                         ['median', 'interpolated', 'collab', 'DAE', 'VAE'],
                                         :]]
metrics.to_csv(FOLDER/ "metrics.csv")
metrics.to_excel(FOLDER/ "metrics.xlsx")
metrics

# %%
level, split = 'NA interpolated', 'valid_fake_na'
selected = metrics.loc[pd.IndexSlice[level,
                          split,
                          :, :]].stack()
selected

# %%
to_plot = selected.loc[level].loc[split].loc[pd.IndexSlice[:,'MAE',:]]
to_plot = to_plot.stack().unstack(REPITITION_NAME).T.describe().loc[['mean','std']].T.unstack(0)
to_plot = to_plot.loc[IDX[0], pd.IndexSlice[:, IDX[1]]]
to_plot.to_csv(FOLDER/ "model_performance_repeated_runs_avg.csv")
to_plot.to_excel(FOLDER/ "model_performance_repeated_runs_avg.xlsx")
to_plot

# %%
logger.setLevel(20) # reset debug
ax = to_plot['mean'].plot.bar(rot=0, width=.8, yerr=to_plot['std'])

# %%
level, split = 'NA interpolated', 'valid_fake_na'
selected = metrics.loc[pd.IndexSlice[level,
                          split,
                          :, 'MAE']].stack(1)
selected.index.names = ('x', 'split', 'model', 'metric', REPITITION_NAME)
# # selected.reset_index()
selected.stack().to_frame('MAE').reset_index()

# %%
fig = sns.barplot(x='data level',
            y='MAE',
            hue='model',
            order = IDX[0],
            ci=95,
            data=selected.stack().to_frame('MAE').reset_index())
fig = ax.get_figure()

# %%
vaep.savefig(fig, FOLDER/ "model_performance_repeated_runs.pdf" )

# %%
