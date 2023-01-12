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
# # Compare predictions between model and RSN
#
# - see differences in imputation for diverging cases
# - dumps top5

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import vaep
import vaep.analyzers
import vaep.io.datasplits
import vaep.imputation

logger = vaep.logging.setup_nb_logger()

# %% [markdown]
# ## Parameters

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
folder_experiment = 'runs/appl_ald_data/plasma/proteinGroups'
model_key = 'vae'
target = 'kleiner'
out_folder = 'diff_analysis'
file_format='pkl'

# %% tags=[]
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args, out_root=args.folder_experiment /
                                 params["out_folder"]/params["target"]/params["model_key"])
args.update_from_dict(params)
args

# %%
files_in= dict(diff_analysis=(args.out_folder /
 f'diff_analysis_compare_methods.xlsx'))
files_in

# %% [markdown]
# ## Load data for different decisions

# %%
differences = pd.read_excel(files_in['diff_analysis'], sheet_name='differences', index_col=[0,1], header=[0,1])
differences[('comp','diff_qvalue')]  = (differences[('RSN', 'qvalue')] - differences[('VAE', 'qvalue')]).abs()
differences = differences.sort_values(('comp','diff_qvalue'), ascending=False)
differences

# %% [markdown]
# ## Measurments

# %%
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y]).unstack()
data

# %%
pred_real_na_imputed_normal = vaep.imputation.impute_shifted_normal(
    df_wide=data)
pred_real_na_imputed_normal = pred_real_na_imputed_normal.unstack()
pred_real_na_imputed_normal

# %%
template = 'pred_real_na_{}.csv'
fname = args.out_preds / template.format(args.model_key)
pred_real_na = vaep.analyzers.compare_predictions.load_single_csv_pred_file(fname)
pred_real_na = pred_real_na.unstack()
pred_real_na.sample(3)

# %%
idx =  differences.index[0]
pg_selected, gene_selected = idx # top feat
pg_selected, gene_selected

# %%
feat_observed = data[pg_selected].dropna()

# %%
# axes = axes.ravel()
folder = args.out_folder / 'intensities_for_diff_in_DA_decision'
folder.mkdir(parents=True, exist_ok=True)
for idx in differences.index:
    pg_selected, gene_selected = idx # top feat
    pg_selected, gene_selected
    fig, ax = plt.subplots()
    dfs = [data[pg_selected].dropna(), pred_real_na[pg_selected].dropna(), pred_real_na_imputed_normal[pg_selected].dropna()]
    
    bins = None
    ax = None
    _series = dfs[0]
    _series_vae = dfs[1]
    _series_rsn = dfs[2]

    ax =_series.hist(ax=ax, bins=bins, label=f'measured (N={len(_series):,d})', color='grey', alpha=0.6)
    ax = _series_vae.hist(ax=ax,bins=bins, label=f'{args.model_key} (N={len(_series_vae):,d})', color='green', alpha=1)
    ax = _series_rsn.hist(ax=ax,bins=bins, label=f'RSN (N={len(_series_rsn):,d})', color='red', alpha=0.8)

    ax.set_title(f'compare imputed vs observed measurments for gene {gene_selected} having a total of {len(data)} samples')
    ax.set_ylabel('count measurments')
    _ = ax.legend()
    vaep.savefig(fig, folder / f'gene_{gene_selected}_pg_{pg_selected.split(";")[0]}.pdf')
