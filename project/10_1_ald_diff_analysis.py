# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# # Differential Analysis - Compare model imputation with standard imputation
#
# - load missing values predictions
# - leave all other values as they were
# - compare missing values predicition by model with baseline method 
#   (default: draw from shifted normal distribution. short RSN)

# %%
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

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

file_format = "csv"
model_key = 'VAE' # model(s) to evaluate
baseline = 'RSN' # default is RSN, but could be any other trained model
value_name='intensity'
out_folder='diff_analysis'
template_pred = 'pred_real_na_{}.csv' # fixed, do not change


# %%
params = vaep.nb.get_params(args, globals=globals(), remove=True)
params

# %%
args = vaep.nb.Config()
args.fn_clinical_data = Path(params["fn_clinical_data"])
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args,
                                 out_root=(args.folder_experiment
                                           / params["out_folder"]
                                           / params["target"]
                                           ))
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

# %%
df_clinic.loc[idx_complete_data, args.target].value_counts()

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


# %% [markdown]
# ## load model predictions for (real) missing data

# %%
list(args.out_preds.iterdir())

# %%
template_pred = str(args.out_preds / args.template_pred)
template_pred

# %%
fname = args.out_preds / args.template_pred.format(args.model_key)
fname 

# %% [markdown]
# Baseline comparison
# In case of RSN -> use filtering as done in original paper (Niu et al. 2022)
# otherwise -> use all data
#
# - use columns which are provided by model
# %%
# ALD study approach -> has access to simulated missing data!
# (VAE model did not see this data)
pred_real_na = None
if args.model_key == 'RSN':
    pred_real_na = vaep.imputation.impute_shifted_normal(
        ald_study)
    pred_real_na.to_csv(fname)
elif args.model_key:
    pred_real_na = (vaep
                    .analyzers
                    .compare_predictions
                    .load_single_csv_pred_file(fname)
                    )
else:
    logger.info('No model key provided -> no imputation of data.')
pred_real_na


# %% [markdown]
# plot subsets to highlight differences


# %%
def plot_distributions(observed: pd.Series,
                       imputation: pd.Series = None,
                       model_key: str = 'MODEL',
                       figsize=(4,3),
                       sharex=True):
    """Plots distributions of intensities provided as dictionary of labels to pd.Series."""
    series_ = [observed, imputation] if imputation is not None else [observed]
    min_bin, max_bin = vaep.plotting.data.get_min_max_iterable(series_)

    if imputation is not None:
        fig, axes = plt.subplots(len(series_), figsize=figsize, sharex=sharex)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, figsize=figsize, sharex=sharex)

    bins = range(min_bin, max_bin+1, 1)
    
    label = 'observed measurments'
    ax = observed.hist(ax=ax, bins=bins, color='grey')
    ax.set_title(f'{label} (N={len(observed):,d})')
    ax.set_ylabel('observations')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")


    if imputation is not None:
        ax = axes[1]
        label = f'Missing values imputed using {model_key.upper()}'
        ax = imputation.hist(ax=ax,bins=bins, color=f'C{1}')
        ax.set_title(f'{label} (N={len(imputation):,d})')
        ax.set_ylabel('observations')
        ax.locator_params(axis='y', integer=True)
        ax.yaxis.set_major_formatter("{x:,.0f}")
    return fig


vaep.plotting.make_large_descriptors(5)
fig = plot_distributions(observed,
                         imputation=pred_real_na,
                         model_key=args.model_key, figsize=(2.5, 2))
fname = args.out_folder / 'dist_plots' / f'real_na_obs_vs_{args.model_key}.pdf'
vaep.savefig(fig, name=fname)

# %% [markdown]
# ## Mean shift by model

# %%
if pred_real_na is not None:
    shifts = (vaep.imputation.compute_moments_shift(observed, pred_real_na,
                                                        names=('observed', args.model_key)))
    display(pd.DataFrame(shifts).T)

# %% [markdown]
# Or by averaging over the calculation by sample

# %%
if pred_real_na is not None:
    index_level = 0  # per sample
    mean_by_sample = pd.DataFrame(
        {'observed': vaep.imputation.stats_by_level(observed, index_level=index_level),
        args.model_key: vaep.imputation.stats_by_level(pred_real_na, index_level=index_level)
    })
    mean_by_sample.loc['mean_shift'] = (mean_by_sample.loc['mean', 'observed'] -
                                        mean_by_sample.loc['mean']).abs() / mean_by_sample.loc['std', 'observed']
    mean_by_sample.loc['std shrinkage'] = mean_by_sample.loc['std'] / \
        mean_by_sample.loc['std', 'observed']
    display(mean_by_sample)

# %% [markdown]
# # Differential analysis
# Impute missing values (or not)

# %%
df = pd.concat([observed, pred_real_na]).unstack()
df.loc[idx_complete_data]

# %%
# if some features were not imputed -> drop them
# could be changed: let a model decide if a feature should be imputed, otherwise don't.
if pred_real_na is not None:
    if df.isna().sum().sum():
        logger.warning("DataFrame has missing entries after imputation.")
        logger.info("Drop columns with missing values.")
    df = df.dropna(axis=1)

# %% [markdown]
# Targets - Clinical variables
# %%
scores = vaep.stats.diff_analysis.analyze(df_proteomics=df,
        df_clinic=df_clinic,
        target=args.target,
        covar=args.covar,
        value_name=args.value_name)

scores.columns = pd.MultiIndex.from_product([[str(args.model_key)], scores.columns],
                                            names=('model', 'var'))
scores.loc[pd.IndexSlice[:, args.target], :]


# %%
fname = args.out_folder/ 'scores' / f'diff_analysis_scores_{str(args.model_key)}.pkl'
fname.parent.mkdir(exist_ok=True, parents=True)
scores.to_pickle(fname)
fname


# %%
list(args.out_folder.iterdir())

# %%
