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
# - load real NA predictions
# - leave all other values as they were
# - compare real NA predicition by model with standard method (draw from shifted normal distribution)

# %%
from pathlib import Path
from collections import namedtuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pingouin as pg

import statsmodels.stats.multitest


import vaep
import vaep.analyzers
import vaep.io.datasplits
import vaep.imputation
import vaep.stats

import vaep.nb as config

logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %%
folder_experiment = "runs/appl_ald_data/proteinGroups"
folder_data: str = ''  # specify data directory if needed
fn_rawfile_metadata = "data/single_datasets/raw_meta.csv"
fn_clinical_data = "data/single_datasets/ald_metadata_cli.csv"
target: str = 'kleiner'
covar:str = 'age,bmi,gender_num,nas_steatosis_ordinal'

file_format = "pkl"
model_key = 'vae'
value_name='intensity'

# %%
params = {k: v for k, v in globals().items() if k not in args and k[0] != '_'}
params

# %%
args = config.Config()
args.fn_rawfile_metadata = Path(fn_rawfile_metadata)
args.fn_clinical_data = Path(fn_clinical_data)
args.folder_experiment = Path(folder_experiment)
args = vaep.nb.add_default_paths(args, folder_data=folder_data)
args.covar = covar.split(',')
args
for k, v in params.items():
    try:
        setattr(args, k, v)
    except AttributeError:
        pass
del fn_rawfile_metadata, fn_clinical_data, folder_experiment, file_format, folder_data, model_key, target, covar
args

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
# covar = ['age', 'bmi', 'gender_num', 'abstinent_num', 'nas_steatosis_ordinal']
# covar_steatosis = ['age', 'bmi', 'gender_num', 'abstinent_num', 'kleiner', 'nas_inflam']

# %%
df_clinic = pd.read_csv(args.fn_clinical_data, index_col=0)
df_clinic = df_clinic.loc[observed.index.levels[0]]
cols_clinic = vaep.pandas.get_columns_accessor(df_clinic)
df_clinic.describe()

# %% [markdown]
# ## ALD study approach using all measurments

# %%
DATA_COMPLETENESS = 0.6
MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622

ald_study, cutoffs = vaep.analyzers.diff_analysis.select_raw_data(observed.unstack(
), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)

ald_study

# %%
freq_feat = observed.unstack().notna().sum()
freq_feat.name = 'frequency'
fname = args.folder_experiment / 'freq_features_observed.csv'
logger.info(fname)
freq_feat.to_csv(fname)
freq_feat

# %%
vaep.plotting.plot_cutoffs(observed.unstack(), feat_completness_over_samples=cutoffs.feat_completness_over_samples,
             min_feat_in_sample=cutoffs.min_feat_in_sample)

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
pred_real_na.sample(3)


# %%
fig, axes = plt.subplots(3, figsize=(10, 15), sharex=True)
ax = axes[1]
ax = pred_real_na.hist(ax=ax)
ax.set_title(f'real na imputed using {args.model_key}')
ax.set_ylabel('count measurments')

ax = axes[0]
ax = observed.hist(ax=ax)
ax.set_title('observed measurments')
ax.set_ylabel('count measurments')

ax = axes[2]
ax = pred_real_na_imputed_normal.hist(ax=ax)
ax.set_title(f'real na imputed using shifted normal distribution')
ax.set_ylabel('count measurments')

vaep.savefig(fig, name=f'real_na_obs_vs_default_vs_{args.model_key}', folder=args.out_figures)

# %% [markdown]
# # Differential analysis

# %% [markdown]
# ## Model imputation

# %%
df = pd.concat([observed, pred_real_na]).unstack()
df

# %%
assert df.isna().sum().sum() == 0, "DataFrame has missing entries"

# %% [markdown]
# Targets - Clinical variables
# %%
scores = vaep.stats.diff_analysis.analyze(df_proteomics=df,
        df_clinic=df_clinic,
        target=args.target,
        covar=args.covar,
        value_name=value_name)

scores.columns = pd.MultiIndex.from_product([[args.model_key], scores.columns],
                                            names=('model', 'var'))
scores


# %% [markdown]
# ## Shifted normal distribution

# %%
df = pd.concat([ald_study.stack(), pred_real_na_imputed_normal]).unstack()
df

# %%
_scores = vaep.stats.diff_analysis.analyze(df_proteomics=df,
        df_clinic=df_clinic,
        target=args.target,
        covar=args.covar,
        value_name=value_name)
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
fname = args.folder_experiment/f'diff_analysis_scores_{args.model_key}.pkl'
scores.to_pickle(fname)
fname
