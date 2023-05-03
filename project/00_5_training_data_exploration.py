# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # # Inspect data using plots
# - spread of intensities between samples
# - spread of intensities within samples
# - missing data plots: violin, box and histogram - both for features and samples
#    - optionally: plot proposed cutoffs (on per default)
# - correlation analysis: can linear correlation be picked up?
# -
#
# Does not save filtered data, this is done by splitting notebook. Only visualisations.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

import vaep
from vaep import plotting
from vaep.pandas import missing_data
import vaep.data_handling
from vaep.analyzers import analyzers

matplotlib.rcParams.update({'font.size': 5,
                            'figure.figsize': [4.0, 2.0]})


# %% [markdown]
# ### Parameters

# %% tags=["parameters"]
FN_INTENSITIES:str = 'data/dev_datasets/df_intensities_proteinGroups_long/Q_Exactive_HF_X_Orbitrap_6070.pkl'
FOLDER_EXPERIMENT:str = 'runs/data_exploration/dev_datasets/proteins/Q_Exactive_HF_X_Orbitrap_6070'
N_FIRST_ROWS = None # possibility to select N first rows
log_transform: bool = True # log transform data
INDEX_COL: list = [0,1] # list of integers or string denoting the index columns (used for csv)
LONG_FORMAT: bool = True # if True, the data is expected to be in long format
# Threshold used later for data filtering (here only for visualisation)
COMPLETENESS_OVER_SAMPLES = 0.25  # 25% of samples have to have that features
MIN_FEAT_PER_SAMPLE = .4  # 40% of features selected in first step


# %%
FOLDER_EXPERIMENT = Path(FOLDER_EXPERIMENT)
FN_INTENSITIES = Path(FN_INTENSITIES)

FIGUREFOLDER = FOLDER_EXPERIMENT / 'figures' 
FIGUREFOLDER.mkdir(exist_ok=True, parents=True)
FIGUREFOLDER

files_out = dict()

# %%
if FN_INTENSITIES.suffix == '.pkl':
    data = pd.read_pickle(FN_INTENSITIES)
elif FN_INTENSITIES.suffix == '.csv':
    data = pd.read_csv(FN_INTENSITIES, INDEX_COL=[0,1], nrows=N_FIRST_ROWS)
data
# %%

# %%
if LONG_FORMAT:
    data = data.unstack()
data = np.log2(data)
data

# %%
records = dict(inital=missing_data.get_record(data))
records

# %% [markdown]
# - filtering based on many other samples?
# - low feature completeness threshold in comparison to other approaches

# %%
# maybe would be good to throw some terrible samples away (hard minimum treshold)
# won't change a lot, but get rid of dummy samples
# mask = data.notna().sum(axis=1) > 200
# mask.sum()

# %%
min_samples_per_feat = int(len(data) * COMPLETENESS_OVER_SAMPLES)
print(f"{min_samples_per_feat = }")
mask = data.notna().sum(axis=0) > min_samples_per_feat
print(f"drop = {(~mask).sum()} features")
selected = data.loc[:, mask]
selected.shape

# %%
min_feat_per_sample = int(selected.shape[-1] * MIN_FEAT_PER_SAMPLE)
print(f"{min_feat_per_sample = }")
samples_selected = selected.notna().sum(axis=1) >= min_feat_per_sample
print(f"drop = {(~samples_selected).sum()} samples")
selected = selected[samples_selected]
selected.shape

# %%
fig = plotting.data.plot_missing_dist_highdim(data,
                                min_feat_per_sample=min_feat_per_sample,
                                min_samples_per_feat=min_samples_per_feat)
fname = FIGUREFOLDER / f'dist_all_lineplot.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)


# %%
fig = plotting.data.plot_missing_pattern_histogram(data,
                               min_feat_per_sample=min_feat_per_sample,
                               min_samples_per_feat=min_samples_per_feat)
fname = FIGUREFOLDER / f'dist_all_histogram.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# Boxplots

# %%
fig = plotting.data.plot_missing_dist_boxplots(data)
fname = FIGUREFOLDER / f'dist_all_boxplots.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %%
fig = plotting.data.plot_missing_pattern_violinplot(
    data, min_feat_per_sample, min_samples_per_feat)
fname = FIGUREFOLDER / f'dist_all_violin_plot.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)


# %%
records.update(
    dict(filtered=missing_data.get_record(selected)))
records.update({'params':
                dict(MIN_FEAT_PER_SAMPLE=float(MIN_FEAT_PER_SAMPLE),
                     COMPLETENESS_OVER_SAMPLES=float(
                         COMPLETENESS_OVER_SAMPLES),
                     min_feat_per_sample=int(min_feat_per_sample),
                     min_samples_per_feat=int(min_samples_per_feat),)
                })
records

# %%
fname = FOLDER_EXPERIMENT / 'records.json'
files_out[fname.name] = fname
with open(fname, 'w') as f:
    json.dump(records, f, indent=4)


# %% [markdown]
# ## Correlation between peptides
# - linear correlation as indicator that there is some variation which could be used by models (or other heuristics)

# %%
# %%time
corr_lower_triangle = analyzers.corr_lower_triangle(data)
fig, axes = analyzers.plot_corr_histogram(corr_lower_triangle, bins=40)
fname = FIGUREFOLDER / f'corr_histogram_feat.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ### Coefficient of variation (CV) of features

# %%
cv = data.std() / data.mean()
# biological coefficient of variation: standard deviation (variation) w.r.t mean
ax = cv.hist(bins=30)
fname = FIGUREFOLDER / f'CV_histogram_features.pdf'
files_out[fname.name] = fname
vaep.savefig(ax.get_figure(), name=fname)


# %% [markdown]
# ## Clustermap and heatmaps of missing values

# %%
# USE_CBAR = False

# axes_heatmap_missing = sns.heatmap(data_to_visualize,
#                                    ax=axes_heatmap_missing,
#                                    cbar = USE_CBAR,
#                                   )

# %% [markdown]
# ## Sample stats

# %%
TYPE = 'feat'
COL_NO_MISSING, COL_NO_IDENTIFIED = f'no_missing_{TYPE}', f'no_identified_{TYPE}'
COL_PROP_SAMPLES = 'prop_samples'


sample_stats = vaep.data_handling.compute_stats_missing(data.notna(), COL_NO_MISSING, COL_NO_IDENTIFIED )
sample_stats

# %%
fig_ident = sns.relplot(
    x='SampleID_int', y=COL_NO_IDENTIFIED, data=sample_stats)
fig_ident.set_axis_labels('Sample ID', f'Frequency of identified {TYPE}')
fig_ident.fig.suptitle(f'Frequency of identified {TYPE} by sample id', y=1.03)
vaep.savefig(fig_ident, f'identified_{TYPE}_by_sample', folder=FIGUREFOLDER)

fig_ident_dist = sns.relplot(
    x=COL_PROP_SAMPLES, y=COL_NO_IDENTIFIED, data=sample_stats)
fig_ident_dist.set_axis_labels(
    'Proportion of samples (sorted by frequency)', f'Frequency of identified {TYPE}')
fig_ident_dist.fig.suptitle(
    f'Frequency of identified {TYPE} groups by sample id', y=1.03)
fname = FIGUREFOLDER / f'identified_{TYPE}_ordered.pdf'
files_out[fname.name] = fname
vaep.savefig(fig_ident_dist, fname)

# %%
COL_NO_MISSING_PROP = COL_NO_MISSING + '_PROP'
sample_stats[COL_NO_MISSING_PROP] = sample_stats[COL_NO_MISSING] / \
    float(data.shape[1])
sns.set(style="white")
g = sns.relplot(x='prop_samples', y=COL_NO_MISSING_PROP, data=sample_stats)
plt.subplots_adjust(top=0.9)
g.set_axis_labels(
    "Proportion of samples (sorted by frequency)", "proportion missing")
g.fig.suptitle(f'Proportion of missing {TYPE} ordered')

fname = FIGUREFOLDER / 'proportion_feat_missing.pdf'
files_out[fname.name] = fname
vaep.savefig(g, fname)

# %% [markdown]
# ### Reference table intensities (log2)

# %%
def get_dynamic_range(min_max):
    dynamic_range = pd.DataFrame(range(*min_max), columns=['x'])
    dynamic_range['$2^x$'] = dynamic_range.x.apply(lambda x: 2**x)
    dynamic_range.set_index('x', inplace=True)
    dynamic_range.index.name = ''
    dynamic_range = dynamic_range.T
    return dynamic_range

min_max = int(data.min().min()), int(data.max().max()) + 1
dynamic_range = None
if min_max[1] < 100:
    dynamic_range = get_dynamic_range(min_max)
dynamic_range

# %%
files_out
