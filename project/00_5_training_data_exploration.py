# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Inspect data using plots
# - spread of intensities between samples
# - spread of intensities within samples
# - missing data plots: violin, box and histogram - both for features and samples
#    - optionally: plot proposed cutoffs (on per default)
# - correlation analysis: can linear correlation be picked up?
# -
#
# Does not save filtered data, this is done by splitting notebook. Only visualisations.


# %%
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import vaep
import vaep.data_handling
from vaep import plotting
from vaep.analyzers import analyzers
from vaep.pandas import missing_data
from vaep.utils import create_random_df

logger = vaep.logging.setup_nb_logger()
logging.getLogger('fontTools').setLevel(logging.WARNING)

matplotlib.rcParams.update({'font.size': 6,
                            'figure.figsize': [4.0, 2.0]})


def get_clustermap(data,
                   figsize=(8, 8),
                   cbar_pos: tuple[float, float, float, float] = (
                       0.02, 0.83, 0.03, 0.15),
                   **kwargs):
    from sklearn.impute import SimpleImputer

    from vaep.pandas import _add_indices
    X = SimpleImputer().fit_transform(data)
    X = _add_indices(X, data)
    cg = sns.clustermap(X,
                        z_score=0,
                        cmap="vlag",
                        center=0,
                        cbar_pos=cbar_pos,
                        figsize=figsize,
                        **kwargs
                        )
    return cg


def get_dynamic_range(min_max):
    dynamic_range = pd.DataFrame(range(*min_max), columns=['x'])
    dynamic_range['$2^x$'] = dynamic_range.x.apply(lambda x: 2**x)
    dynamic_range.set_index('x', inplace=True)
    dynamic_range.index.name = ''
    dynamic_range = dynamic_range.T
    return dynamic_range


# %% [markdown]
# Expected current format:
# - wide format (samples x features)
# > not the default output in MS-based proteomics
#
# An example of peptides in wide format would be:

# %%
create_random_df(5, 8, prop_na=.2)

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
FN_INTENSITIES: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
FOLDER_EXPERIMENT: str = 'runs/example/data_inspection'
N_FIRST_ROWS = None  # possibility to select N first rows
LOG_TRANSFORM: bool = True  # log transform data
# list of integers or string denoting the index columns (used for csv)
INDEX_COL: list = [0]
COL_INDEX_NAME: str = 'Protein groups'  # name of column index, can be None
LONG_FORMAT: bool = False  # if True, the data is expected to be in long format
# Threshold used later for data filtering (here only for visualisation)
COMPLETENESS_OVER_SAMPLES = 0.25  # 25% of samples have to have that features
MIN_FEAT_PER_SAMPLE = .4  # 40% of features selected in first step
# protein group separator, e.g.';'  (could also be gene groups)
PG_SEPARATOR: str = ';'
SAMPLE_FIRST_N_CHARS: int = 16  # number of characters used for sample names
# if True, do not use tick on heatmap - only label
NO_TICK_LABELS_ON_HEATMAP: bool = True
FEATURES_CUTOFF: int = 10_000  # cutoff for number of features to plot in clustermaps or heatmaps, randomly selected


# %% [markdown]
# ## Load and check data
#
# - supported for now: pickle and comma separated
# - transform long to wide data?
# - log transform data using logarithm of two?
# - remove entirely missing columns (features) or rows (samples)

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
    data = pd.read_csv(FN_INTENSITIES, index_col=INDEX_COL, nrows=N_FIRST_ROWS)
elif FN_INTENSITIES.suffix == '.tsv':
    data = pd.read_csv(FN_INTENSITIES, sep='\t', index_col=INDEX_COL, nrows=N_FIRST_ROWS)
else:
    raise ValueError(f'File extension {FN_INTENSITIES.suffix} not supported')
data

# %%
if LONG_FORMAT:
    data = data.squeeze().unstack()
if LOG_TRANSFORM:
    data = np.log2(data).astype(float)


# drop entrily missing rows or columns
data = data.dropna(axis=0, how='all').dropna(axis=1, how='all')
data

# %%
if len(data.columns.names) > 1:
    _levels_dropped = data.columns.names[1:]
    data.columns = data.columns.droplevel(_levels_dropped)
    logger.warning("Drop multiindex level, kepp only first. Dropped: "
                   f"{_levels_dropped}")
# allows overwriting of index name, also to None
data.columns.name = COL_INDEX_NAME
data


# %% [markdown]
# ## Calculate cutoffs for visualization and stats

# %% [markdown]
# - filtering based on many other samples?
# - low feature completeness threshold in comparison to other approaches

# %%
min_samples_per_feat = int(len(data) * COMPLETENESS_OVER_SAMPLES)
print(f"{min_samples_per_feat = }")
mask = data.notna().sum(axis=0) >= min_samples_per_feat
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

# %% [markdown]
# ### Update records if cutoffs would be used

# %%
records = dict(inital=missing_data.get_record(data))
records

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
del selected  # try to free memory (in case a copy was created in pandas 3.0)
records

# %%
fname = FOLDER_EXPERIMENT / 'records.json'
files_out[fname.name] = fname
with open(fname, 'w') as f:
    json.dump(records, f, indent=4)


# %% [markdown]
# ## Plot basic distribution present-absent pattern of features and samples
#
# ### Line plots

# %%
fig = plotting.data.plot_missing_dist_highdim(data,
                                              min_feat_per_sample=min_feat_per_sample,
                                              min_samples_per_feat=min_samples_per_feat)
fname = FIGUREFOLDER / 'dist_all_lineplot_w_cutoffs.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %%
fig = plotting.data.plot_missing_dist_highdim(data)
fname = FIGUREFOLDER / 'dist_all_lineplot_wo_cutoffs.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %%
fig = plotting.data.plot_missing_pattern_histogram(data,
                                                   min_feat_per_sample=min_feat_per_sample,
                                                   min_samples_per_feat=min_samples_per_feat)
fname = FIGUREFOLDER / 'dist_all_histogram_w_cutoffs.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %%
fig = plotting.data.plot_missing_pattern_histogram(data)
fname = FIGUREFOLDER / 'dist_all_histogram_wo_cutoffs.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ### Boxplots

# %%
fig = plotting.data.plot_missing_dist_boxplots(data)
fname = FIGUREFOLDER / 'dist_all_boxplots.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ### Violinplots

# %%
fig = plotting.data.plot_missing_pattern_violinplot(
    data, min_feat_per_sample, min_samples_per_feat)
fname = FIGUREFOLDER / 'dist_all_violin_plot.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ## Feature medians over prop. of missing of feature
# %%
ax = plotting.data.plot_feat_median_over_prop_missing(
    data=data, type='scatter', s=1)
fname = FIGUREFOLDER / 'intensity_median_vs_prop_missing_scatter'
files_out[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = plotting.data.plot_feat_median_over_prop_missing(
    data=data, type='boxplot', s=.8)
fname = FIGUREFOLDER / 'intensity_median_vs_prop_missing_boxplot'
files_out[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)


# %% [markdown]
# ## Correlation between peptides
# - linear correlation as indicator that there is some variation which could be used by models (or other heuristics)

# %%
# %%time
if data.shape[1] > FEATURES_CUTOFF:
    selected = data.sample(n=FEATURES_CUTOFF, axis=1, random_state=42)
    FEATURES_CUTOFF_TEXT = f'{FEATURES_CUTOFF:,d} randomly selected {COL_INDEX_NAME}'
else:
    FEATURES_CUTOFF = data.shape[1]
    FEATURES_CUTOFF_TEXT = f'{FEATURES_CUTOFF:,d} {COL_INDEX_NAME}'
    selected = data
FEATURES_CUTOFF_TEXT

# %%
corr_lower_triangle = analyzers.corr_lower_triangle(selected)
fig, axes = analyzers.plot_corr_histogram(corr_lower_triangle, bins=40)
fig.suptitle(f'Histogram of correlations based on {FEATURES_CUTOFF_TEXT}')
fname = FIGUREFOLDER / 'corr_histogram_feat.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)


# %% [markdown]
# ### Coefficient of variation (CV) of features

# %%
cv = data.std() / data.mean()
# biological coefficient of variation: standard deviation (variation) w.r.t mean
ax = cv.hist(bins=30)
ax.set_title(f'Histogram of coefficient of variation (CV) of {FEATURES_CUTOFF_TEXT}')
fname = FIGUREFOLDER / 'CV_histogram_features.pdf'
files_out[fname.name] = fname
vaep.savefig(ax.get_figure(), name=fname)

# %% [markdown]
# ## Clustermap and heatmaps of missing values

# %%
# needs to deal with duplicates
# notna = data.notna().T.drop_duplicates().T
# get index and column names
vaep.plotting.make_large_descriptors(5)

cg = sns.clustermap(selected.notna(),
                    cbar_pos=None,
                    figsize=(8, 8))
ax = cg.ax_heatmap
if PG_SEPARATOR is not None:
    _new_labels = [_l.get_text().split(PG_SEPARATOR)[0]
                   for _l in ax.get_xticklabels()]
    _ = ax.set_xticklabels(_new_labels)
if NO_TICK_LABELS_ON_HEATMAP:
    ax.set_xticks([])
    ax.set_yticks([])
# cg.fig.suptitle(f'Present-absent pattern of {FEATURES_CUTOFF_TEXT}')
ax.set_title(f'Present-absent pattern of {FEATURES_CUTOFF_TEXT}')
cg.figure.tight_layout()
fname = FIGUREFOLDER / 'clustermap_present_absent_pattern.png'
files_out[fname.name] = fname
vaep.savefig(cg.figure,
             name=fname,
             pdf=False,
             dpi=600)

# %% [markdown]
# based on cluster, plot heatmaps of features and samples

# %%
assert (len(cg.dendrogram_row.reordered_ind), len(
    cg.dendrogram_col.reordered_ind)) == selected.shape

# %%
vaep.plotting.make_large_descriptors(5)
fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax = sns.heatmap(
    selected.iloc[cg.dendrogram_row.reordered_ind,
                  cg.dendrogram_col.reordered_ind],
    robust=True,
    cbar=False,
    annot=False,
    ax=ax,
)
ax.set_title(f'Heatmap of intensities clustered by missing pattern of {FEATURES_CUTOFF_TEXT}',
             fontsize=8)
vaep.plotting.only_every_x_ticks(ax, x=2)
vaep.plotting.use_first_n_chars_in_labels(ax, x=SAMPLE_FIRST_N_CHARS)
if PG_SEPARATOR is not None:
    _new_labels = [_l.get_text().split(PG_SEPARATOR)[0]
                   for _l in ax.get_xticklabels()]
    _ = ax.set_xticklabels(_new_labels)
if NO_TICK_LABELS_ON_HEATMAP:
    ax.set_xticks([])
    ax.set_yticks([])
fname = FIGUREFOLDER / 'heatmap_intensities_ordered_by_missing_pattern.png'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname, pdf=False, dpi=600)
# ax.get_figure().savefig(fname, dpi=300)

# %% [markdown]
# ### Heatmap of sample and feature correlation

# %%
fig, ax = plt.subplots(figsize=(4, 4))
ax = sns.heatmap(
    analyzers.corr_lower_triangle(
        selected.iloc[:, cg.dendrogram_col.reordered_ind]),
    vmin=-1,
    vmax=1,
    cbar_kws={'shrink': 0.75},
    ax=ax,
    square=True,
)
ax.set_title(f'Heatmap of feature correlation of {FEATURES_CUTOFF_TEXT}',
             fontsize=8)
_ = vaep.plotting.only_every_x_ticks(ax, x=2)
_ = vaep.plotting.use_first_n_chars_in_labels(ax, x=SAMPLE_FIRST_N_CHARS)
if PG_SEPARATOR is not None:
    _new_labels = [_l.get_text().split(PG_SEPARATOR)[0]
                   for _l in ax.get_xticklabels()]
    _ = ax.set_xticklabels(_new_labels)
if NO_TICK_LABELS_ON_HEATMAP:
    ax.set_xticks([])
    ax.set_yticks([])
fname = FIGUREFOLDER / 'heatmap_feature_correlation.png'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname, pdf=False, dpi=600)

# %%
lower_corr = analyzers.corr_lower_triangle(
    selected.T.iloc[:, cg.dendrogram_row.reordered_ind])

# %%
fig, ax = plt.subplots(figsize=(4, 4))
ax = sns.heatmap(
    data=lower_corr,
    ax=ax,
    vmin=-1,
    vmax=1,
    cbar_kws={'shrink': 0.75},
    square=True,
)
_ = vaep.plotting.only_every_x_ticks(ax, x=2)
_ = vaep.plotting.use_first_n_chars_in_labels(ax, x=SAMPLE_FIRST_N_CHARS)
if NO_TICK_LABELS_ON_HEATMAP:
    ax.set_xticks([])
    ax.set_yticks([])
ax.set_title(f'Heatmap of sample correlation based on {FEATURES_CUTOFF_TEXT}', fontsize=7)
fname = FIGUREFOLDER / 'heatmap_sample_correlation.png'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname, pdf=False, dpi=600)

# %%
vaep.plotting.make_large_descriptors(6)
kwargs = dict()
if NO_TICK_LABELS_ON_HEATMAP:
    kwargs['xticklabels'] = False
    kwargs['yticklabels'] = False
cg = get_clustermap(selected, **kwargs)
ax = cg.ax_heatmap
if PG_SEPARATOR is not None:
    _new_labels = [_l.get_text().split(PG_SEPARATOR)[0]
                   for _l in ax.get_xticklabels()]
    _ = ax.set_xticklabels(_new_labels)
_ = vaep.plotting.only_every_x_ticks(ax, x=2, axis=0)
_ = vaep.plotting.use_first_n_chars_in_labels(ax, x=SAMPLE_FIRST_N_CHARS)
# ax.set_title(f'Clustermap of intensities based on {FEATURES_CUTOFF_TEXT}', fontsize=7)
# cg.fig.tight_layout()  # tight_layout makes the cbar a bit ugly
cg.fig.suptitle(f'Clustermap of intensities based on {FEATURES_CUTOFF_TEXT}', fontsize=7)
fname = FIGUREFOLDER / 'clustermap_intensities_normalized.png'
files_out[fname.name] = fname
cg.fig.savefig(fname, dpi=300)  # avoid tight_layout
# vaep.savefig(cg.fig,
#              name=fname,
#              pdf=False)

# %% [markdown]
# ## Sample stats

# %%
TYPE = 'feat'
COL_NO_MISSING, COL_NO_IDENTIFIED = f'no_missing_{TYPE}', f'no_identified_{TYPE}'
COL_PROP_SAMPLES = 'prop_samples'

sample_stats = vaep.data_handling.compute_stats_missing(
    data.notna(), COL_NO_MISSING, COL_NO_IDENTIFIED)
sample_stats

# %%
vaep.plotting.make_large_descriptors(8)
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
plt.ylim(0, 1)
g.set_axis_labels(
    "Proportion of samples (sorted by frequency)", "proportion missing")
g.fig.suptitle(f'Proportion of missing {TYPE} ordered')

fname = FIGUREFOLDER / 'proportion_feat_missing.pdf'
files_out[fname.name] = fname
vaep.savefig(g, fname)

# %% [markdown]
# ### Reference table intensities (log2)

# %%
min_max = int(data.min().min()), int(data.max().max()) + 1
dynamic_range = None
if min_max[1] < 100:
    dynamic_range = get_dynamic_range(min_max)
dynamic_range


# %%
files_out

# %%
