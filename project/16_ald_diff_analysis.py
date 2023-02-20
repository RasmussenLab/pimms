# ---
# jupyter:
#   jupytext:
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
# # Differential Analysis - Compare model imputation with standard imputation
#
# - load missing values predictions
# - leave all other values as they were
# - compare missing values predicition by model with standard method (draw from shifted normal distribution. short RSN)

# %%
from pathlib import Path
from collections import namedtuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pingouin as pg


import vaep
import vaep.analyzers
import vaep.io.datasplits
import vaep.imputation
import vaep.stats

import vaep.nb

logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"
folder_data: str = ''  # specify data directory if needed
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"
fn_qc_samples = '' #'data/ALD_study/processed/qc_plasma_proteinGroups.pkl'


target: str = 'kleiner'
covar:str = 'age,bmi,gender_num,nas_steatosis_ordinal,abstinent_num'

file_format = "pkl"
model_key = 'vae'
value_name='intensity'
out_folder='diff_analysis'

# %%
params = vaep.nb.get_params(args, globals=globals(), remove=True)
params

# %%
args = vaep.nb.Config()
args.fn_clinical_data = Path(params["fn_clinical_data"])
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args, out_root=args.folder_experiment/params["out_folder"]/params["target"]/params["model_key"])
args.covar = params["covar"].split(',')
args.update_from_dict(params)
args

# %% [markdown]
# Outputs of this notebook will be stored here

# %%
args.out_folder

# %% [markdown]
# # Data

# %% [markdown]
# ## MS proteomics

# %%
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)

# %%
observed = pd.concat([data.train_X, data.val_y, data.test_y])
observed

# %% [markdown]
# ## Clinical data

# %%
df_clinic = pd.read_csv(args.fn_clinical_data, index_col=0)
df_clinic = df_clinic.loc[observed.index.levels[0]]
cols_clinic = vaep.pandas.get_columns_accessor(df_clinic) # pick Berlin as reference?
df_clinic[[args.target, *args.covar]].describe()

# %% [markdown]
# Entries with missing values
# - see how many rows have one missing values (for target and covariates)
# - only complete data is used for Differential Analysis
# - covariates are not imputed

# %%
df_clinic[[args.target, *args.covar]].isna().any(axis=1).sum()

# %% [markdown]
# Data description of data used:

# %%
idx_complete_data = df_clinic[[args.target, *args.covar]].dropna().index
df_clinic.loc[idx_complete_data, [args.target, *args.covar]].describe()

# %% [markdown]
# ## ALD study approach using all measurments

# %%
DATA_COMPLETENESS = 0.6
# MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622
CV_QC_SAMPLE: float = 0.4 # Coef. of variation on 13 QC samples

ald_study, cutoffs = vaep.analyzers.diff_analysis.select_raw_data(observed.unstack(
), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)

ald_study

# %%
if args.fn_qc_samples:
    # Move this to data-preprocessing
    qc_samples = pd.read_pickle(args.fn_qc_samples)
    qc_cv_feat = qc_samples.std() / qc_samples.mean()
    qc_cv_feat = qc_cv_feat.rename(qc_samples.columns.name)
    fig, ax = plt.subplots(figsize=(4,7))
    ax = qc_cv_feat.plot.box(ax=ax)
    ax.set_ylabel('Coefficient of Variation')
    vaep.savefig(fig, name='cv_qc_samples', folder=args.out_figures)
    print((qc_cv_feat < CV_QC_SAMPLE).value_counts())
    # only to ald_study data
    ald_study = ald_study[vaep.analyzers.diff_analysis.select_feat(qc_samples[ald_study.columns])]
    
ald_study

# %%
freq_feat = observed.unstack().notna().sum()
freq_feat.name = 'frequency'
fname = args.folder_experiment / 'freq_features_observed.csv'
logger.info(fname)
freq_feat.to_csv(fname)
freq_feat

# %%
fig, axes = vaep.plotting.plot_cutoffs(observed.unstack(),
             feat_completness_over_samples=cutoffs.feat_completness_over_samples,
             min_feat_in_sample=cutoffs.min_feat_in_sample)
vaep.savefig(fig, name='tresholds_normal_imputation', folder=args.out_figures)

# %%
pred_real_na_imputed_normal = vaep.imputation.impute_shifted_normal(
    ald_study)

# %% [markdown]
# ## load model predictions for (real) missing data

# %%
list(args.out_preds.iterdir())

# %%
template = 'pred_real_na_{}.csv'
fname = args.out_preds / template.format(args.model_key)
fname

# %%
pred_real_na = vaep.analyzers.compare_predictions.load_single_csv_pred_file(fname)
pred_real_na.sample(3).to_frame()


# %%
min_bin, max_bin = (int(min(pred_real_na.min(), observed.min(), pred_real_na_imputed_normal.min())),
(int(max(pred_real_na.max(), observed.max(), pred_real_na_imputed_normal.max()))) + 1)
min_bin, max_bin

# %%
fig, axes = plt.subplots(3, figsize=(10, 10), sharex=True)

# axes = axes.ravel()
bins = range(min_bin, max_bin+1, 1)
ax = axes[0]
ax = observed.hist(ax=ax, bins=bins, color='grey')
ax.set_title(f'observed measurments (N={len(observed):,d})')
ax.set_ylabel('count measurments')

ax = axes[1]
ax = pred_real_na.hist(ax=ax,bins=bins, label=f'all (N={len(pred_real_na):,d})')
ax.set_title(f'Missing values imputed using {args.model_key} (N={len(pred_real_na):,d})')
ax.set_ylabel('count measurments')

idx_new_model = pred_real_na.index.difference(pred_real_na_imputed_normal.index)
ax = pred_real_na.loc[idx_new_model].hist(ax=ax,bins=bins, label=f'new (N={len(idx_new_model):,d})', color='green', alpha=0.9)
ax.legend()

ax = axes[2]
ax = pred_real_na.loc[pred_real_na_imputed_normal.index].hist(ax=ax,bins=bins, label=args.model_key)
ax = pred_real_na_imputed_normal.hist(ax=ax, bins=bins, label='RSN')

ax.set_title(f'missing values imputed by RSN (N={len(pred_real_na_imputed_normal):,d})')
ax.set_ylabel('count measurments')
ax.set_xlabel(args.value_name)
ax.legend(fontsize='xx-large')

vaep.savefig(fig, name=f'real_na_obs_vs_default_vs_{args.model_key}_v2', folder=args.out_folder)

# %% [markdown]
# plot subsets to highlight differences

# %%
fig, axes = plt.subplots(3, figsize=(10, 10), sharex=True)

ax = axes[0]
ax = observed.hist(ax=ax, bins=bins, color='grey')
ax.set_title(f'observed measurments (N={len(observed):,d})')
ax.set_ylabel('count measurments')

ax = axes[1]
bins = range(min_bin, max_bin+1, 1)
ax = pred_real_na.hist(ax=ax,bins=bins)
ax.set_title(f'Missing values imputed using {args.model_key} (N={len(pred_real_na):,d})')
ax.set_ylabel('count measurments')

ax = axes[2]
ax = pred_real_na_imputed_normal.hist(ax=ax, bins=bins, color='C1')
ax.set_title(f'Missing values imputed using RSN (N={len(pred_real_na_imputed_normal):,d})')
ax.set_ylabel('count measurments')
ax.set_xlabel(args.value_name)

vaep.savefig(fig, name=f'real_na_obs_vs_default_vs_{args.model_key}', folder=args.out_folder)

# %% [markdown]
# ## Mean shift by method

# %%
observed.mean(), observed.std(), pred_real_na.mean(), pred_real_na.std(), pred_real_na_imputed_normal.mean(), pred_real_na_imputed_normal.std()

# %%
shifts = vaep.imputation.compute_moments_shift(observed, pred_real_na_imputed_normal, names=('observed', 'RSN'))
shifts.update(vaep.imputation.compute_moments_shift(observed, pred_real_na, names=('observed', args.model_key)))
pd.DataFrame(shifts).T

# %% [markdown]
# Or by averaging over the calculation by sample

# %%
index_level = 0  # per sample
mean_by_sample = {}
mean_by_sample['observed'] = vaep.imputation.stats_by_level(
    observed, index_level=index_level)
mean_by_sample['RSN'] = vaep.imputation.stats_by_level(
    pred_real_na_imputed_normal, index_level=index_level)
mean_by_sample['vae'] = vaep.imputation.stats_by_level(
    pred_real_na, index_level=index_level)
mean_by_sample = pd.DataFrame(mean_by_sample)
mean_by_sample.loc['mean_shift'] = (mean_by_sample.loc['mean', 'observed'] -
                                    mean_by_sample.loc['mean']).abs() / mean_by_sample.loc['std', 'observed']
mean_by_sample.loc['std shrinkage'] = mean_by_sample.loc['std'] / \
    mean_by_sample.loc['std', 'observed']
mean_by_sample

# %% [markdown]
# # Differential analysis

# %% [markdown]
# ## Model imputation

# %%
df = pd.concat([observed, pred_real_na]).unstack()
df.loc[idx_complete_data]

# %%
assert df.isna().sum().sum() == 0, "DataFrame has missing entries"

# %% [markdown]
# Targets - Clinical variables
# %%
scores = vaep.stats.diff_analysis.analyze(df_proteomics=df,
        df_clinic=df_clinic,
        target=args.target,
        covar=args.covar,
        value_name=args.value_name)

scores.columns = pd.MultiIndex.from_product([[args.model_key], scores.columns],
                                            names=('model', 'var'))
scores


# %% [markdown]
# ## Shifted normal distribution
# - select protein groups which were in original study
# - loaded from config

# %%
df = pd.concat([ald_study.stack(), pred_real_na_imputed_normal]).unstack()
ald_study_feat = df.columns.to_list()
df = df.loc[idx_complete_data]
df

# %%
_scores = vaep.stats.diff_analysis.analyze(df_proteomics=df,
        df_clinic=df_clinic,
        target=args.target,
        covar=args.covar,
        value_name=args.value_name)
_scores.columns = pd.MultiIndex.from_product([['random shifted_imputation'], _scores.columns],
                                            names=('model', 'var'))
_scores
# %% [markdown]
# # Combine scores

# %%
scores=scores.join(_scores)

# %%
scores.describe()

# %%
fname = args.out_folder/f'diff_analysis_scores.pkl'
scores.to_pickle(fname)
fname

# %% [markdown]
# ## Save new features with target for further use

# %%
df = pd.concat([observed, pred_real_na]).unstack()
df = df[df.columns.difference(ald_study_feat)]
df = df.join(df_clinic[args.target]).dropna()
df.to_pickle(args.out_folder / f'new_features.pkl')
df

# %%
list(args.out_folder.iterdir())
