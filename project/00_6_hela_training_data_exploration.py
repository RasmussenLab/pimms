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

# %% [markdown] Collapsed="false"
# # Peptides
#
# Load peptides selected for training

# %% Collapsed="false"
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn import preprocessing
# from sklearn.decomposition import PCA
import seaborn as sns

import vaep
from vaep.data_handling import coverage
from vaep.plotting import _savefig

import config
from vaep.analyzers import analyzers
from vaep.io.data_objects import PeptideCounter

pd.options.display.max_columns = 100
pd.options.display.min_rows = 30

# %% [markdown]
# ## Descriptive Statistics (Linear case)
#
# - spread of peptide quantifications between samples
# - spread of quantifications within samples
# - correlation analysis: can linear correlation be picked up?
#

# %% [markdown]
# ### Peptides

# %%
FN_PEPTIDE_INTENSITIES = Path('data/dev_datasets/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.csv')
FIGUREFOLDER = FN_PEPTIDE_INTENSITIES.parent / 'figures' / FN_PEPTIDE_INTENSITIES.stem
FIGUREFOLDER.mkdir(exist_ok=True, parents=True)
FIGUREFOLDER

# %%
N_FIRST_ROWS = None # possibility to select N first rows
analysis = analyzers.AnalyzePeptides.from_csv(fname=FN_PEPTIDE_INTENSITIES, index_col=[0,1],nrows=N_FIRST_ROWS)
df = analysis.to_wide_format()
analysis.describe_peptides(sample_n=30)

# %% [markdown]
# ### Peptide frequency: sellect the N most common peptides
#
# - N most common peptides between samples

# %%
N = 10

peptide_counter = PeptideCounter(config.FNAME_C_PEPTIDES)
peptide_counter.counter.most_common(N)

# %%
counts = analysis.df.count().sort_values(ascending=False)
counts.iloc[:N]

# %%
analysis.df[counts.iloc[:N].index]

# %% [markdown]
# ## Correlation between peptides
# - linear correlation as indicator that there is some variation which could be used by models (or other heuristics)

# %%
sample = analysis.df.sample(n=30, axis=1)
# ToDo func is assigned to df
corr_lower_triangle = analyzers.corr_lower_triangle(sample)
corr_lower_triangle

# %%
fig, axes = analyzers.plot_corr_histogram(corr_lower_triangle, bins=40)

# %% [markdown]
# ### Samples

# %%
analysis.df.sample(30, axis=0).T.describe()

# %% [markdown]
# ### Peptides (all)

# %%
stats = analysis.describe_peptides()

# %%
_ = stats.loc['CV'].hist(figsize=(10, 4)) # biological coefficient of variation: standard deviation (variation) w.r.t mean

# %%
_ = stats.loc['count'].hist(figsize=(10,4))

# %% Collapsed="false"
INDEX_NAME = 'Sample ID'
analysis.df.index.name = INDEX_NAME

# %% Collapsed="false"
analysis.df

# %% Collapsed="false"
N_MIN_OBS = analysis.df.shape[0] * 0.7 # here: present in 70% of the samples
mask_min_obsevation = analysis.df.notna().sum() >= N_MIN_OBS
mask_min_obsevation.sum()

# %% [markdown]
# Reference analysis.df as `X`

# %%
X = analysis.df

# %% [markdown] Collapsed="false"
# ## Completeness of peptides

# %% Collapsed="false"
# %time not_missing = vaep.data_handling.get_sorted_not_missing(X)
not_missing.iloc[:, -10:].describe()

# %% Collapsed="false"
sample_completeness = not_missing.sum(axis=1).sort_values() / X.shape[-1]
sample_completeness

# %% Collapsed="false"
N_MOST_COMMON_PEPTIDES = 300
data_to_visualize = not_missing.iloc[:, -N_MOST_COMMON_PEPTIDES:]
data_to_visualize = data_to_visualize.loc[sample_completeness.index]
print(f"Look at missingness pattern of {N_MOST_COMMON_PEPTIDES} most common peptides across sample.\n"
      f"Data matrix dimension used for printing: { data_to_visualize.shape}")


fig_heatmap_missing, axes_heatmap_missing = plt.subplots(
    1, 1, figsize=(12, 8))
USE_CBAR = False

axes_heatmap_missing = sns.heatmap(data_to_visualize,
                                   ax=axes_heatmap_missing,
                                   cbar = USE_CBAR,
                                  )

# %% [markdown]
# White patches indicates that a peptide has been measured, black means it was not measured. Some samples (rows) have few of the most common peptides. This suggests to set a minimum of total peptides in a sample, which is common pratice. 
#
# > An algorithm should work with the most common peptides and base it's inference capabilities after training on these.

# %%
data_to_visualize.sum(axis=1).nsmallest(20) # Samplest with the fewest measurements out of the seletion

# %% Collapsed="false"
# # This currently crashes if you want to have a pdf
datetime_now = datetime.now()
_savefig = partial(_savefig, folder=FIGUREFOLDER)

_savefig(fig_heatmap_missing,
         f'peptides_heatmap_missing_{datetime_now:%y%m%d}', pdf=False)

# %% [markdown] Collapsed="false"
# ## Sample stats

# %% Collapsed="false"
TYPE = 'peptides'
COL_NO_MISSING, COL_NO_IDENTIFIED = f'no_missing_{TYPE}', f'no_identified_{TYPE}'
COL_PROP_SAMPLES = 'prop_samples'


sample_stats = vaep.data_handling.compute_stats_missing(not_missing, COL_NO_MISSING, COL_NO_IDENTIFIED )
sample_stats

# %% Collapsed="false"
fig_ident = sns.relplot(
    x='SampleID_int', y=COL_NO_IDENTIFIED, data=sample_stats)
fig_ident.set_axis_labels('Sample ID', f'Frequency of identified {TYPE}')
fig_ident.fig.suptitle(f'Frequency of identified {TYPE} by sample id', y=1.03)
_savefig(fig_ident, f'identified_{TYPE}_by_sample', folder=FIGUREFOLDER)

fig_ident_dist = sns.relplot(
    x=COL_PROP_SAMPLES, y=COL_NO_IDENTIFIED, data=sample_stats)
fig_ident_dist.set_axis_labels(
    'Proportion of samples (sorted by frequency)', f'Frequency of identified {TYPE}')
fig_ident_dist.fig.suptitle(
    f'Frequency of identified {TYPE} groups by sample id', y=1.03)
_savefig(fig_ident_dist, f'identified_{TYPE}_ordered', folder=FIGUREFOLDER)

# %% Collapsed="false"
COL_NO_MISSING_PROP = COL_NO_MISSING + '_PROP'
sample_stats[COL_NO_MISSING_PROP] = sample_stats[COL_NO_MISSING] / \
    float(X.shape[1])

# from ggplot import *
# ggplot(aes(x='nan_proc'), data = nonnan) + geom_histogram(binwidth = 0.005) #+ ylim(0,0.025)
sns.set(style="darkgrid")
g = sns.relplot(x='prop_samples', y=COL_NO_MISSING_PROP, data=sample_stats)
plt.subplots_adjust(top=0.9)
g.set_axis_labels(
    "Proportion of samples (sorted by frequency)", "proportion missing")
g.fig.suptitle(f'Proportion of missing {TYPE} ordered')
_savefig(g, "proportion_proteins_missing")


# %% [markdown] Collapsed="false"
# ## Look at sequences
#
# Shows mainly that from a 6-7 AA on, peptides sequences are nearly unique.
#
# > Overlapping peptides (from the start or the end) could still be interesting to find

# %% Collapsed="false"
class SequenceAnalyser():

    def __init__(self, sequences: pd.Series):
        if not isinstance(sequences, pd.Series):
            raise ValueError(
                "Please provide a pandas.Series, not {}".format(type(sequences)))
        self.sequences = sequences

    def calc_counts(self, n_characters):
        return self.sequences.str[:n_characters].value_counts()

    def length(self):
        return self.sequences.str.len().sort_values()


# %% Collapsed="false"
sequences = SequenceAnalyser(X.columns.to_series())
sequences.length()

# %% Collapsed="false"
import ipywidgets as w
_ = w.interact(sequences.calc_counts,
           n_characters=w.IntSlider(value=4, min=1, max=55))

# %% Collapsed="false"
sequences_p4 = sequences.calc_counts(4)
display(sequences_p4.head())

# %% Collapsed="false"
sequences_p4.loc[sequences_p4.isin(('CON_', 'REV_'))].sort_index()

# %% [markdown] Collapsed="false"
# What to do when 
#
#
# ```
# AAAAAAAAAAGAAGGRGSGPGR
# AAAAAAAAAAGAAGGRGSGPGRR
#
# AAAANSGSSLPLFDCPTWAGKPPPGLHLDVVK
# AAAANSGSSLPLFDCPTWAGKPPPGLHLDVVKGDK
# ```
#
#

# %% [markdown] Collapsed="false"
# ## Select Training Data

# %% [markdown] Collapsed="false"
# ### Minumum required sample quality
# First define the minum requirement of a sample to be kept in 

# %% Collapsed="false"
import ipywidgets as w
range_peps = (0,  max(sample_stats[COL_NO_IDENTIFIED]))
MIN_DEPTH_SAMPLE = int(range_peps[1] * 0.6)
w_min_depth_sample = w.IntSlider(
    value=MIN_DEPTH_SAMPLE, min=0, max=range_peps[1])
print(f'Minimum {TYPE} per sample observed:')
w_min_depth_sample

# %% Collapsed="false"
mask_samples = sample_stats[COL_NO_IDENTIFIED] >= w_min_depth_sample.value
print(f"Selected {mask_samples.sum()} samples")

# %% Collapsed="false"
x_50 = coverage(X.loc[mask_samples], coverage_col=0.5, coverage_row=0.2)
# x_50_pca = log_z_zeroone_na(x_50) # there is a huge difference if NA is set to low value or mean!!
x_90 = coverage(X.loc[mask_samples], 0.9, 0.9)

# %% Collapsed="false"
x_50.shape, x_90.shape

# %% Collapsed="false"
x_90.sample()

# %% [markdown]
# Data selection should be done for each experiment, so it is not resaved here

# %%
#from vaep.io.data_objects import get_fname
# fname = config.FOLDER_DATA / get_fname(*x_90.shape)
# print(fname)
# x_90.to_csv(fname)
# fname = config.FOLDER_DATA / get_fname(*x_50.shape)
# print(fname)
# x_50.to_csv(fname)

# %% [markdown] Collapsed="false"
# ### Distribution of Intensity values
# - comparing non-transformed to $\log_{10}$ transformed
# - log transformation makes data more normal distributed
#
# > log10 or log2 or ln

# %% [markdown]
# #### Sample with all peptides

# %% Collapsed="false"
sample = x_50.sample().iloc[0]
sample_id = sample.name 
print("Sample ID:", sample_id)

# %% Collapsed="false"
import matplotlib

sns.set(style="darkgrid")


def plot_dist_comparison(
    sample: pd.Series, figsize=(12, 5),
    log=np.log, log_name=None,
) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.histplot(sample, bins=100, ax=axes[0])
    axes[0].set_title("Unnormalized distribution")

    sample_log = log(sample)
    sns.histplot(sample_log, bins=100, ax=axes[1])
    if not log_name:
        log_name = str(log).split("'")[1]
    axes[1].set_title(f"{log_name} normalized distribution")
    sample_id = sample.name
    _ = fig.suptitle(f"Dynamic Range of measured intensities in sample {sample_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


fig = plot_dist_comparison(sample)
_savefig(fig, f"distribution_sample_peptides_{str(sample_id)}_ln")

# %%
fig = plot_dist_comparison(sample, log=np.log2)
_savefig(fig, f"distribution_peptides_sample_{str(sample_id)}_log2")

# %%
sample_log_stats       = np.log2(sample).describe().to_frame('log2')
sample_log_stats['ln'] = np.log (sample).describe()
sample_log_stats

# %%
print(f"Factor for log2 to ln: {1 / np.log2(np.e) = :.3f}")
c = 1 / np.log2(np.e)

# %% [markdown]
# If $ log2(x) \sim \mathcal{N}\big(\mu_{log2}, \sigma_{log2}^2 \big) $, then $ ln(x) \sim \mathcal{N}\big(0.693 \cdot \mu_{log2}, 0.693^2 \cdot \sigma_{log2}^2 \big) $.
#
# > Question: Is a wider or narrower distribtion important, or does only be "normal"

# %%
print(f"mean: {sample_log_stats.loc['mean','log2'] * c = : .3f}")
print(f"std : {sample_log_stats.loc['std' ,'log2'] * c = : .3f}")

# %% [markdown]
# #### One Peptide, all samples

# %% Collapsed="false"
from vaep.transform import log
from random import sample
sample = x_50.sample(axis=1).squeeze()
peptide = sample.name

fig = plot_dist_comparison(sample)
_savefig(fig, f"distribution_peptide_samples_{str(peptide)}_ln")

# %% [markdown] Collapsed="false"
# ### Reference table intensities (natural logarithm)
#
# 14 to 23 spans a dynamic range of 3 orders of base 10

# %% Collapsed="false"
dynamic_range = pd.DataFrame(range(14, 24), columns=['x'])
dynamic_range['$e^x$'] = dynamic_range.x.apply(np.exp)
dynamic_range.set_index('x', inplace=True)
dynamic_range.index.name = ''
dynamic_range.T

# %% [markdown] Collapsed="false"
# ## Next UP

# %% [markdown]
#

# %% [markdown] Collapsed="false"
# ### Find Protein of Peptides
# - check with some reference list of peptides: This is created in `project\FASTA_tryptic_digest.ipynb` 

# %%
