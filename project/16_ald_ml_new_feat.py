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
folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"
folder_data: str = ''  # specify data directory if needed
fn_clinical_data = "data/single_datasets/ald_metadata_cli.csv"

folder_experiment = "runs/appl_ald_data/plasma/proteinGroups"
model_key = 'vae'
target = 'kleiner'
sample_id_col = 'Sample ID'
cutoff_target:int = 2 # => for binarization target >= cutoff_target
file_format = "pkl"
out_folder='diff_analysis'

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
# ## Load new features

# %% tags=[]
target = pd.read_csv(args.fn_clinical_data, index_col=0, usecols=[args.sample_id_col, args.target])
target = target.dropna()
target

# %%
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y])
# data.loc[df_clinic.index]

# %%
in_both = data.index.levels[0].intersection(target.index)
assert not in_both.empty, f"No shared indices: {data.index.levels[0]} and {target.index}"
print(f"Samples available both in proteomics data and for target: {len(in_both)}")
target, data = target.loc[in_both], data.loc[in_both]

# %%
DATA_COMPLETENESS = 0.6
MIN_N_PROTEIN_GROUPS: int = 200
FRAC_PROTEIN_GROUPS: int = 0.622

ald_study, cutoffs = vaep.analyzers.diff_analysis.select_raw_data(data.unstack(
), data_completeness=DATA_COMPLETENESS, frac_protein_groups=FRAC_PROTEIN_GROUPS)

ald_study

# %%
template = 'pred_real_na_{}.csv'
fname = args.out_preds / template.format(args.model_key)
print(f"REAL NA pred. by {args.model_key}: {fname}")
pred_real_na = vaep.analyzers.compare_predictions.load_single_csv_pred_file(fname).loc[in_both]
pred_real_na.sample(3)

# %%
pred_real_na_imputed_normal = vaep.imputation.impute_shifted_normal(ald_study)

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
# ## Train, test split

# %%
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, target, stratify=target, random_state=42)
idx_train = X_train.index
idx_test = X_test.index

# %%
vaep.pandas.combine_value_counts(pd.concat([y_train, y_test], axis=1, ignore_index=True
                                           ).rename(columns={0: 'train', 1: 'test'})
                                 )

# %% [markdown]
# Binarize targets (stay the same for all three configurations of data to be tested)

# %%
y_train = y_train >= args.cutoff_target
y_test = y_test >= args.cutoff_target

# %%
vaep.pandas.combine_value_counts(pd.concat([y_train, y_test], axis=1, ignore_index=True
                                           ).rename(columns={0: 'train', 1: 'test'})
                                 )

# %% [markdown]
# ## Results
#
# - `run_model` returns dataclasses with the further needed results

# %%
splits = Splits(X_train=X.loc[idx_train], X_test=X.loc[idx_test], y_train=y_train, y_test=y_test)
results_model_full = vaep.sklearn.run_model(splits)
results_model_full.name = f'{args.model_key} all'

# %%
splits = Splits(X_train=X.loc[idx_train, new_features], X_test=X.loc[idx_test, new_features], y_train=y_train, y_test=y_test)
results_model_new = vaep.sklearn.run_model(splits)
results_model_new.name = f'{args.model_key} new'

# %%
splits_ald = Splits(X_train=ald_study.loc[idx_train], X_test=ald_study.loc[idx_test], y_train=y_train, y_test=y_test)
results_ald_full = vaep.sklearn.run_model(splits_ald)
results_ald_full.name = 'ALD study all'

# %% [markdown]
# - plot X_train PCA, map X_test

# %%
fig, ax = plt.subplots(1,1)
plot_split_auc(results_ald_full.test, results_ald_full.name, ax)
plot_split_auc(results_model_full.test, results_model_full.name, ax)
plot_split_auc(results_model_new.test, results_model_new.name, ax)
vaep.savefig(fig, name='auc_roc_curve', folder=args.out_folder)

# %%
selected_features = pd.DataFrame([results_ald_full.selected_features, results_model_full.selected_features, results_model_new.selected_features], index=[results_ald_full.name, results_model_full.name, results_model_new.name]).T
selected_features.index.name = 'rank'
selected_features.to_excel(args.out_folder / 'mrmr_feat_by_model.xlsx')
selected_features

# %%
col_name = f'full model (auc: {results_model_full.test.auc:.3f})'
roc = pd.DataFrame(results_model_full.test.roc, index='fpr tpr cutoffs'.split()).rename({'tpr': col_name})
ax = roc.T.plot('fpr', col_name, ylabel='tpr', ax=ax)

# %%
fig, ax = plt.subplots(1,1)

ax = plot_split_prc(results_ald_full.test, results_ald_full.name, ax)
ax = plot_split_prc(results_model_full.test, results_model_full.name, ax)
ax = plot_split_prc(results_model_new.test, results_model_new.name, ax)
vaep.savefig(fig, name='prec_recall_curve', folder=args.out_folder)
