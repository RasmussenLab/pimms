# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import yaml
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
FOLDER = Path('runs/mnar_mcar/')
SIZE = 'l'
files_in = {
    'protein groups': FOLDER / 'pg_l_25MNAR/figures/2_1_performance_test_sel.csv',
    'peptides': FOLDER / 'pep_l_25MNAR/figures/2_1_performance_test.csv',
    'precursors': FOLDER / 'evi_l_25MNAR/figures/2_1_performance_test.csv'
}

# %%
FOLDER = Path('runs/mnar_mcar/')
SIZE = 'm'
files_in = {
    'protein groups': FOLDER / 'pg_m_25MNAR/figures/2_1_performance_test_sel.csv',
    'peptides': FOLDER / 'pep_m_25MNAR/figures/2_1_performance_test_sel.csv',
    'precursors': FOLDER / 'evi_m_25MNAR/figures/2_1_performance_test_sel.csv'
}

# %%
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

# %% [markdown]
# color mapping globally defined for article figures

# %%
COLORS_TO_USE_MAPPTING = vaep.plotting.defaults.color_model_mapping
print(COLORS_TO_USE_MAPPTING.keys())
sns.color_palette(palette=COLORS_TO_USE_MAPPTING.values())

# %%
data_levels_annotated = dict()
for key, fname in files_in.items():
    fname = fname.parents[1] / 'data_config.yaml'
    with open(fname) as f:
        data_config = yaml.safe_load(f)
    data_levels_annotated[key] = f"{key} \n (N={data_config['N']:,d}, M={data_config['M']:,d})"
    # print(pd.read_csv(file).mean())
# data_levels_annotated
ORDER_DATA = list(data_levels_annotated.values())
df = df.rename(index=data_levels_annotated)
df

# %%
fname = FOLDER / f'best_models_{SIZE}_test_mpl.pdf'
metrics = df['metric_value'].unstack('model')
ORDER_MODELS = metrics.mean().sort_values().index.to_list()
metrics = metrics.loc[ORDER_DATA, ORDER_MODELS]

plt.rcParams['figure.figsize'] = [4.0, 2.0]
matplotlib.rcParams.update({'font.size': 6})

ax = (metrics
      .plot
      .bar(rot=0,
           xlabel='',
           ylabel=f"{METRIC} (log2 intensities)",
           color=COLORS_TO_USE_MAPPTING,
           width=.85,
           fontsize=7
           ))


ax = vaep.plotting.add_height_to_barplot(ax, size=6, rotated=True)
ax.set_ylim((0, 0.75))
ax.legend(fontsize=5, loc='lower right')
text = (
    df['text']
    .unstack()
    .fillna('')
    .stack().loc[pd.IndexSlice[ORDER_MODELS, ORDER_DATA]]

)
ax = vaep.plotting.add_text_to_barplot(ax, text, size=6)
fig = ax.get_figure()
fig.tight_layout()
vaep.savefig(fig, fname)


# %%
df = metrics.fillna(0.0).stack().to_frame(
    'metric_value').join(text.rename('text'))
df.to_excel(fname.with_suffix('.xlsx'))

# %% [markdown]
# # aggregate all mean results

# %%
files_perf = {k: f.parent.parent /
              '01_2_performance_summary.xlsx' for k, f in files_in.items()}
files_perf

# %%
perf = dict()
for k, f in files_perf.items():
    df = pd.read_excel(f, index_col=0, sheet_name=1)
    perf[(k, 'val')] = df.loc['mean']
    df = pd.read_excel(f, index_col=0, sheet_name=2)
    perf[(k, 'test')] = df.loc['mean']

perf = pd.DataFrame(perf)
order = (perf
         .loc[:, pd.IndexSlice[:, 'val']]
         .mean(axis=1)
         .sort_values()
         .index)
perf = perf.loc[order]
perf

# %%
fname = FOLDER / f'performance_summary_{SIZE}.xlsx'
perf.to_excel(fname)
fname.as_posix()

# %%
