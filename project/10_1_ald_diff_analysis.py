# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
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
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import njab.stats
import pandas as pd
from IPython.display import display

import vaep
import vaep.analyzers
import vaep.imputation
import vaep.io.datasplits
import vaep.nb

logger = vaep.logging.setup_nb_logger()
logging.getLogger('fontTools').setLevel(logging.WARNING)

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
fn_qc_samples = ''  # 'data/ALD_study/processed/qc_plasma_proteinGroups.pkl'
f_annotations = 'data/ALD_study/processed/ald_plasma_proteinGroups_id_mappings.csv'


target: str = 'kleiner'
covar: str = 'age,bmi,gender_num,nas_steatosis_ordinal,abstinent_num'

file_format = "csv"
model_key = 'VAE'  # model(s) to evaluate
model = None  # default same as model_key, but could be overwritten (edge case)
value_name = 'intensity'
out_folder = 'diff_analysis'
template_pred = 'pred_real_na_{}.csv'  # fixed, do not change


# %%
if not model:
    model = model_key
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
files_out = {}
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
cols_clinic = vaep.pandas.get_columns_accessor(df_clinic)
df_clinic[[args.target, *args.covar]].describe()


# %%
# ## Additional annotations
# - additional annotations of features (e.g. gene names for protein groups)

feat_name = observed.index.names[-1]
if args.f_annotations:
    gene_to_PG = pd.read_csv(args.f_annotations)
    gene_to_PG = gene_to_PG.drop_duplicates().set_index(feat_name)
    fname = args.out_folder / Path(args.f_annotations).name
    gene_to_PG.to_csv(fname)
    files_out[fname.name] = fname.as_posix()
else:
    gene_to_PG = None
gene_to_PG

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
mask_sample_with_complete_clinical_data = df_clinic[[args.target, *args.covar]].notna().all(axis=1)
fname = args.out_folder / 'mask_sample_with_complete_clinical_data.csv'
files_out[fname.name] = fname.as_posix()
mask_sample_with_complete_clinical_data.to_csv(fname)

idx_complete_data = (mask_sample_with_complete_clinical_data
                     .loc[mask_sample_with_complete_clinical_data]
                     .index)
df_clinic.loc[idx_complete_data, [args.target, *args.covar]].describe()

# %%
df_clinic.loc[idx_complete_data, args.target].value_counts()

# %% [markdown]
# check which patients with kleiner score have misssing covariates

# %%
df_clinic.loc[(~mask_sample_with_complete_clinical_data
               & df_clinic[args.target].notna()),
              [args.target, *args.covar]]

# %% [markdown]
# Save feature frequency of observed data based on complete clinical data

# %%
feat_freq_observed = observed.unstack().loc[idx_complete_data].notna().sum()
feat_freq_observed.name = 'frequency'

fname = args.folder_experiment / 'freq_features_observed.csv'
files_out['feat_freq_observed'] = fname.as_posix()
logger.info(fname)
feat_freq_observed.to_csv(fname)
ax = feat_freq_observed.sort_values().plot(marker='.', rot=90)
_ = ax.set_xticklabels([l_.get_text().split(';')[0] for l_ in ax.get_xticklabels()])

# %% [markdown]
# ## ALD study approach using all measurments

# %%
DATA_COMPLETENESS = 0.6
# MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622
CV_QC_SAMPLE: float = 0.4  # Coef. of variation on 13 QC samples

ald_study, cutoffs = vaep.analyzers.diff_analysis.select_raw_data(observed.unstack(
), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)

ald_study

# %%
if args.fn_qc_samples:
    # Move this to data-preprocessing
    qc_samples = pd.read_pickle(args.fn_qc_samples)
    qc_cv_feat = qc_samples.std() / qc_samples.mean()
    qc_cv_feat = qc_cv_feat.rename(qc_samples.columns.name)
    fig, ax = plt.subplots(figsize=(4, 7))
    ax = qc_cv_feat.plot.box(ax=ax)
    ax.set_ylabel('Coefficient of Variation')
    vaep.savefig(fig, name='cv_qc_samples', folder=args.out_figures)
    print((qc_cv_feat < CV_QC_SAMPLE).value_counts())
    # only to ald_study data
    ald_study = ald_study[vaep.analyzers.diff_analysis.select_feat(qc_samples[ald_study.columns])]

ald_study

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
fname = args.out_preds / args.template_pred.format(args.model)
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
if args.model_key and str(args.model_key) != 'None':
    pred_real_na = (vaep
                    .analyzers
                    .compare_predictions
                    .load_single_csv_pred_file(fname)
                    )
else:
    logger.info('No model key provided -> no imputation of data.')

if args.model_key == 'RSN':
    logger.info('Filtering of data as done in original paper for RSN.')
    # Select only idx from RSN which are selected by ald study cutoffs
    idx_to_sel = ald_study.columns.intersection(pred_real_na.index.levels[-1])
    pred_real_na = pred_real_na.loc[pd.IndexSlice[:, idx_to_sel]]
pred_real_na


# %% [markdown]
# plot subsets to highlight differences


# %%
def plot_distributions(observed: pd.Series,
                       imputation: pd.Series = None,
                       model_key: str = 'MODEL',
                       figsize=(4, 3),
                       sharex=True):
    """Plots distributions of intensities provided as dictionary of labels to pd.Series."""
    series_ = [observed, imputation] if imputation is not None else [observed]
    min_bin, max_bin = vaep.plotting.data.get_min_max_iterable([observed])

    if imputation is not None:
        fig, axes = plt.subplots(len(series_), figsize=figsize, sharex=sharex)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, figsize=figsize, sharex=sharex)

    bins = range(min_bin, max_bin + 1, 1)

    label = 'observed measurments'
    ax = observed.hist(ax=ax, bins=bins, color='grey')
    ax.set_title(f'{label} (N={len(observed):,d})')
    ax.set_ylabel('observations')
    ax.locator_params(axis='y', integer=True)
    ax.yaxis.set_major_formatter("{x:,.0f}")

    if imputation is not None:
        ax = axes[1]
        label = f'Missing values imputed using {model_key.upper()}'
        color = vaep.plotting.defaults.color_model_mapping.get(model_key, None)
        if color is None:
            color = f'C{1}'
        ax = imputation.hist(ax=ax, bins=bins, color=color)
        ax.set_title(f'{label} (N={len(imputation):,d})')
        ax.set_ylabel('observations')
        ax.locator_params(axis='y', integer=True)
        ax.yaxis.set_major_formatter("{x:,.0f}")
    return fig


vaep.plotting.make_large_descriptors(6)
fig = plot_distributions(observed,
                         imputation=pred_real_na,
                         model_key=args.model_key, figsize=(2.5, 2))
fname = args.out_folder / 'dist_plots' / f'real_na_obs_vs_{args.model_key}.pdf'
files_out[fname.name] = fname.as_posix()
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
# * if some features were not imputed -> drop them
# ? could be changed: let a model decide if a feature should be imputed, otherwise don't.
if pred_real_na is not None:
    if df.isna().sum().sum():
        logger.warning("DataFrame has missing entries after imputation.")
        logger.info("Drop columns with missing values.")
    df = df.dropna(axis=1)

# %% [markdown]
# Targets - Clinical variables

# %%
scores = njab.stats.ancova.AncovaAll(df_proteomics=df,
                                     df_clinic=df_clinic,
                                     target=args.target,
                                     covar=args.covar,
                                     value_name=args.value_name
                                     ).ancova()
scores

# %%
# features are in first index position
feat_idx = scores.index.get_level_values(0)
if gene_to_PG is not None:
    scores = (scores
              .join(gene_to_PG)
              .set_index(gene_to_PG.columns.to_list(), append=True)
              )
scores

# %%
scores.columns = pd.MultiIndex.from_product([[str(args.model_key)], scores.columns],
                                            names=('model', 'var'))
scores.loc[pd.IndexSlice[:, args.target], :]


# %%
fname = args.out_folder / 'scores' / f'diff_analysis_scores_{str(args.model_key)}.pkl'
files_out[fname.name] = fname.as_posix()
fname.parent.mkdir(exist_ok=True, parents=True)
scores.to_pickle(fname)
fname


# %%
files_out
# %%
