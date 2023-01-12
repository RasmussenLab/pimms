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
# # Compare outcomes from differential analysis based on different imputation methods
#
# - load scores based on `16_ald_diff_analysis`

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

import vaep
import vaep.analyzers
import vaep.imputation
import vaep.io.datasplits


import vaep.sklearn
from vaep.sklearn.types import Splits
from vaep.plotting.metrics import plot_split_auc, plot_split_prc


logger = vaep.logging.setup_nb_logger()

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
folder_data: str = ''  # specify data directory if needed
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"
folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"
model_key = 'vae'
target = 'kleiner'
sample_id_col = 'Sample ID'
cutoff_target:int = 2 # => for binarization target >= cutoff_target
file_format = "pkl"
out_folder='diff_analysis'
fn_qc_samples = 'data/ALD_study/processed/qc_plasma_proteinGroups.pkl'

# %%
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args, out_root=args.folder_experiment/params["out_folder"]/params["target"]/params["model_key"])
args.update_from_dict(params)
args

# %% [markdown]
# ## Load target

# %% tags=[]
target = pd.read_csv(args.fn_clinical_data, index_col=0, usecols=[args.sample_id_col, args.target])
target = target.dropna()
target

# %% [markdown]
# ### Measured data

# %%
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y])
data.sample(5)

# %% [markdown]
# Get overlap between independent features and target

# %%
in_both = data.index.levels[0].intersection(target.index)
assert not in_both.empty, f"No shared indices: {data.index.levels[0]} and {target.index}"
print(f"Samples available both in proteomics data and for target: {len(in_both)}")
target, data = target.loc[in_both], data.loc[in_both]

# %% [markdown]
# ### Load ALD data or create

# %%
DATA_COMPLETENESS = 0.6
MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622
CV_QC_SAMPLE: float = 0.4

ald_study, cutoffs = vaep.analyzers.diff_analysis.select_raw_data(data.unstack(
), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)

if args.fn_qc_samples:
    qc_samples = pd.read_pickle(args.fn_qc_samples)
    qc_samples = qc_samples[ald_study.columns]
    qc_cv_feat = qc_samples.std() / qc_samples.mean()
    qc_cv_feat = qc_cv_feat.rename(qc_samples.columns.name)
    fig, ax = plt.subplots(figsize=(4,7))
    ax = qc_cv_feat.plot.box(ax=ax)
    ax.set_ylabel('Coefficient of Variation')
    print((qc_cv_feat < CV_QC_SAMPLE).value_counts())
    ald_study = ald_study[vaep.analyzers.diff_analysis.select_feat(qc_samples)]

column_name_first_prot_to_pg = {pg.split(';')[0]: pg for pg in data.unstack().columns}

ald_study = ald_study.rename(columns=column_name_first_prot_to_pg)
ald_study

# %% [markdown]
# ### Load semi-supervised model imputations

# %%
template = 'pred_real_na_{}.csv'
fname = args.out_preds / template.format(args.model_key)
print(f"REAL NA pred. by {args.model_key}: {fname}")
pred_real_na = vaep.analyzers.compare_predictions.load_single_csv_pred_file(fname).loc[in_both]
pred_real_na.sample(3)

# %%
pred_real_na_imputed_normal = vaep.imputation.impute_shifted_normal(ald_study)
pred_real_na_imputed_normal

# %% [markdown]
# # Model predictions
#
# General approach:
#   - use one train, test split of the data
#   - select best 10 features from training data `X_train`, `y_train` before binarization of target
#   - dichotomize (binarize) data into to groups (zero and 1)
#   - evaluate model on the test data `X_test`, `y_test`
#  
# Repeat general approach for
#  1. all original ald data: all features justed in original ALD study
#  2. all model data: all features available my using the self supervised deep learning model
#  3. newly available feat only: the subset of features available from the self supervised deep learning model which were newly retained using the new approach

# %%
X = pd.concat([data, pred_real_na]).unstack()
X

# %%
ald_study = pd.concat([ald_study.stack(), pred_real_na_imputed_normal]).unstack()
ald_study

# %%
new_features = X.columns.difference(ald_study.columns)
new_features

# %% [markdown]
# Binarize targets, but also keep groups for stratification
#

# %%
target_to_group = target.copy()
target = target >= args.cutoff_target
pd.crosstab(target.squeeze(), target_to_group.squeeze())

# %% [markdown]
# ## Best number of parameters by CV

# %%
cv_feat_ald = vaep.sklearn.find_n_best_features(X=ald_study, y=target, name=args.target,
                                                groups=target_to_group)
cv_feat_ald = cv_feat_ald.groupby('n_features').agg(['mean', 'std'])
cv_feat_ald

# %%
cv_feat_all = vaep.sklearn.find_n_best_features(X=X, y=target, name=args.target,
                                               groups=target_to_group)
cv_feat_all = cv_feat_all.groupby('n_features').agg(['mean', 'std'])
cv_feat_all

# %%
cv_feat_new = vaep.sklearn.find_n_best_features(X=X.loc[:, new_features],
                                                y=target, name=args.target,
                                                groups=target_to_group)
cv_feat_new = cv_feat_new.groupby('n_features').agg(['mean', 'std'])
cv_feat_new

# %%
n_feat_best = pd.DataFrame({'ald': cv_feat_ald.loc[:, pd.IndexSlice[:,'mean']].idxmax(),
 'all': cv_feat_all.loc[:, pd.IndexSlice[:,'mean']].idxmax(),
 'new': cv_feat_new.loc[:, pd.IndexSlice[:,'mean']].idxmax()}).droplevel(-1)
n_feat_best

# %% [markdown]
# ## Train, test split

# %%
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, target, test_size=.2,
    stratify=target_to_group, random_state=42)
idx_train = X_train.index
idx_test = X_test.index

# %%
vaep.pandas.combine_value_counts(pd.concat([y_train, y_test], axis=1, ignore_index=True
                                           ).rename(columns={0: 'train', 1: 'test'})
                                 )

# %%
# y_train = y_train >= args.cutoff_target
# y_test = y_test >= args.cutoff_target

# %%
vaep.pandas.combine_value_counts(pd.concat([y_train, y_test], axis=1, ignore_index=True
                                           ).rename(columns={0: 'train', 1: 'test'})
                                 )

# %%
y_train.value_counts()

# %% [markdown]
# ## Results
#
# - `run_model` returns dataclasses with the further needed results
# - add mrmr selection of data (select best number of features to use instead of fixing it)

# %%
splits = Splits(X_train=X.loc[idx_train], X_test=X.loc[idx_test], y_train=y_train, y_test=y_test)
results_model_full = vaep.sklearn.run_model(splits, n_feat_to_select=n_feat_best.loc['test_roc_auc', 'all'])
results_model_full.name = f'{args.model_key} all'

# %%
splits = Splits(X_train=X.loc[idx_train, new_features], X_test=X.loc[idx_test, new_features], y_train=y_train, y_test=y_test)
results_model_new = vaep.sklearn.run_model(splits,  n_feat_to_select=n_feat_best.loc['test_roc_auc', 'new'])
results_model_new.name = f'{args.model_key} new'

# %%
splits_ald = Splits(X_train=ald_study.loc[idx_train], X_test=ald_study.loc[idx_test], y_train=y_train, y_test=y_test)
results_ald_full = vaep.sklearn.run_model(splits_ald,  n_feat_to_select=n_feat_best.loc['test_roc_auc', 'ald'])
results_ald_full.name = 'ALD study all'

# %% [markdown]
# ### ROC-AUC

# %%
figsize=(8,8)
fig, ax = plt.subplots(1,1, figsize=figsize)
plot_split_auc(results_ald_full.test, results_ald_full.name, ax)
plot_split_auc(results_model_full.test, results_model_full.name, ax)
plot_split_auc(results_model_new.test, results_model_new.name, ax)
vaep.savefig(fig, name='auc_roc_curve', folder=args.out_folder)

# %% [markdown]
# ### Features selected

# %%
selected_features = pd.DataFrame([results_ald_full.selected_features, results_model_full.selected_features, results_model_new.selected_features], index=[results_ald_full.name, results_model_full.name, results_model_new.name]).T
selected_features.index.name = 'rank'
selected_features.to_excel(args.out_folder / 'mrmr_feat_by_model.xlsx')
selected_features

# %% [markdown]
# ### Precision-Recall plot

# %%
fig, ax = plt.subplots(1,1, figsize=figsize)

ax = plot_split_prc(results_ald_full.test, results_ald_full.name, ax)
ax = plot_split_prc(results_model_full.test, results_model_full.name, ax)
ax = plot_split_prc(results_model_new.test, results_model_new.name, ax)
vaep.savefig(fig, name='prec_recall_curve', folder=args.out_folder)

# %% [markdown]
# ## Train data plots

# %%
fig, ax = plt.subplots(1,1, figsize=figsize)

ax = plot_split_prc(results_ald_full.train, results_ald_full.name, ax)
ax = plot_split_prc(results_model_full.train, results_model_full.name, ax)
ax = plot_split_prc(results_model_new.train, results_model_new.name, ax)
vaep.savefig(fig, name='prec_recall_curve_train', folder=args.out_folder)

# %%
figsize=(10,7)
fig, ax = plt.subplots(1,1, figsize=figsize)
plot_split_auc(results_ald_full.train, results_ald_full.name, ax)
plot_split_auc(results_model_full.train, results_model_full.name, ax)
plot_split_auc(results_model_new.train, results_model_new.name, ax)
vaep.savefig(fig, name='auc_roc_curve_train', folder=args.out_folder)

# %% [markdown]
# Options:
# - F1 results for test data for best cutoff on training data? 
#   (select best cutoff of training data, evaluate on test data)
# - plot X_train PCA/UMAP, map X_test
