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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import vaep.plotting
import vaep.pandas
import vaep.nb

import logging
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'), level=10)



# %%
# parameters
FOLDER = Path('runs/Q_Exactive_HF_X_Orbitrap_6070/')
files_in = {
      'protein groups': FOLDER / 'proteinGroups/figures/performance_test.csv',
      'peptides': FOLDER / 'peptides/figures/performance_test.csv',
      'precursors': FOLDER / 'evidence/figures/performance_test.csv'
}

# %%
# FOLDER = Path('runs/dev_dataset_small/')
# files_in = {
#       'protein groups': Path('runs/example') / 'figures/performance_test.csv',
#       'peptides': FOLDER / 'peptides_N50/figures/performance_test.csv',
#       'precursors': FOLDER / 'evidence_N50/figures/performance_test.csv'
# }

# %%
ORDER_DATA = list(files_in.keys())
METRIC = 'MAE'

# %%
df = list()
for key, file_in in files_in.items():
    _ = pd.read_csv(file_in)
    _['data level'] = key
    df.append(_)
df = pd.concat(df, axis=0)
df.columns = ['model', *df.columns[1:]]
df = df.set_index(list(df.columns[:2]))
df

# %%
fname = FOLDER / 'best_models_1_test_mpl.pdf'
metrics = df['metric_value'].unstack('model')
ORDER_MODELS = metrics.mean().sort_values().index.to_list()
metrics = metrics.loc[ORDER_DATA, ORDER_MODELS]

plt.rcParams['figure.figsize'] = [4.0, 3.0]
matplotlib.rcParams.update({'font.size': 5})

ax = (metrics
      .plot
      .bar(rot=0,
            xlabel='',
            ylabel=f"{METRIC} (log2 intensities)",
            # position=0.1,
            width=.85))

ax = vaep.plotting.add_height_to_barplot(ax, size=5)
ax.legend(fontsize=5, loc='lower right')
text = (
    df['text']
    .unstack()
    .fillna('')
    .stack().loc[pd.IndexSlice[ORDER_MODELS, ORDER_DATA]]

)
ax = vaep.plotting.add_text_to_barplot(ax, text, size=5)
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, fname)


# %%
df = metrics.fillna(0.0).stack().to_frame('metric_value').join(text.rename('text'))
df.to_excel(fname.with_suffix('.xlsx'))
