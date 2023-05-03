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
# # Compare predictions between model and RSN
#
# - see differences in imputation for diverging cases
# - dumps top5

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn

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
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"
model_key = 'VAE'
sample_id_col = 'Sample ID'
target = 'kleiner'
cutoff_target: int = 2  # => for binarization target >= cutoff_target
out_folder = 'diff_analysis'
file_format = 'pkl'
baseline = 'RSN' # default is RSN, but could be any other trained model
template_pred = 'pred_real_na_{}.csv' # fixed, do not change


# %%
folder_experiment = 'runs/appl_ald_data_old/plasma/proteinGroups'

# %%
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args,
                                 out_root=(args.folder_experiment
                                           / params["out_folder"]
                                           / params["target"]
                                           / f"{params['baseline']}_vs_{params['model_key']}"))
args.update_from_dict(params)
args

# %%
files_in = dict(diff_analysis=(args.out_folder /
                               f'diff_analysis_compare_methods.xlsx'))
files_in

# %% [markdown]
# ## Load data for different decisions

# %%
# blank index columns -> previous entry is used..
differences = pd.read_excel(
    files_in['diff_analysis'], sheet_name='differences', index_col=[0, 1], header=[0, 1])
differences[('comp', 'diff_qvalue')] = (
    differences[(args.baseline, 'qvalue')] - differences[(args.model_key, 'qvalue')]).abs()
differences = differences.sort_values(('comp', 'diff_qvalue'), ascending=False)
differences

# %% [markdown]
# ## Load target

# %%
target = pd.read_csv(args.fn_clinical_data, index_col=0,
                     usecols=[args.sample_id_col, args.target])
target = target.dropna()
target

# %%
target_to_group = target.copy()
target = target >= args.cutoff_target
pd.crosstab(target.squeeze(), target_to_group.squeeze())

# %% [markdown]
# ## Measurments

# %%
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y]).unstack()
data


# %% [markdown]
# better load RSN prediction
# - RSN prediction are based on all samples mean and std (N=455) as in original study
# - VAE also trained on all samples (self supervised)
# One could also reduce the selected data to only the samples with a valid target marker,
# but this was not done in the original study which considered several different target markers.
#
# RSN : shifted per sample, not per feature!
# %%
# reload
pred_real_na_imputed_baseline = vaep.imputation.impute_shifted_normal(
    df_wide=data)
pred_real_na_imputed_baseline = pred_real_na_imputed_baseline.unstack()
pred_real_na_imputed_baseline

# %%
fname = args.out_preds / args.template_pred.format(args.model_key)
pred_real_na = vaep.analyzers.compare_predictions.load_single_csv_pred_file(
    fname)
pred_real_na = pred_real_na.unstack()
pred_real_na.sample(3)

# %% [markdown]
# Once imputation, reduce to target samples only (samples with target score)

# %%
# select samples with target information
data = data.loc[target.index]
pred_real_na_imputed_baseline = pred_real_na_imputed_baseline.loc[target.index]
pred_real_na = pred_real_na.loc[target.index]

assert len(data) == len(pred_real_na) == len(pred_real_na_imputed_baseline)


# %%
idx = differences.index[0]
pg_selected, gene_selected = idx  # top feat
pg_selected, gene_selected

# %%
feat_observed = data[pg_selected].dropna()

# %%
# axes = axes.ravel()
folder = args.out_folder / 'intensities_for_diff_in_DA_decision'
folder.mkdir(parents=True, exist_ok=True)


# %%
min_y_int, max_y_int = vaep.plotting.data.get_min_max_iterable(
    [data.stack(), pred_real_na.stack(), pred_real_na_imputed_baseline.stack()])
min_max = min_y_int, max_y_int

target_name = target.columns[0]

min_max, target_name

# %%
for idx in differences.index:
    pg_selected, gene_selected = idx  # top feat
    fig, ax = plt.subplots()
    dfs = [data[pg_selected].dropna(), pred_real_na[pg_selected].dropna(),
           pred_real_na_imputed_baseline[pg_selected].dropna()]

    bins = None
    ax = None
    _series = dfs[0]
    _series_vae = dfs[1]
    _series_rsn = dfs[2]

    ax, bins = vaep.plotting.data.plot_histogram_intensites(
        _series,
        ax=ax,
        min_max=min_max,
        label=f'measured (N={len(_series):,d})',
        color='grey',
        alpha=0.6)
    ax, bins = vaep.plotting.data.plot_histogram_intensites(
        _series_vae,
        ax=ax,
        min_max=min_max,
        label=f'{args.model_key.upper()} (N={len(_series_vae):,d})',
        color='green',
        alpha=1)
    ax, bins = vaep.plotting.data.plot_histogram_intensites(
        _series_rsn,
        ax=ax,
        min_max=min_max,
        label=f'{args.baseline.upper()} (N={len(_series_rsn):,d})',
        color='red',
        alpha=0.8)

    ax.set_title(
        f'Imputation for protein group {pg_selected.split(";")[0]} (gene: {gene_selected}) with target {target_name} (N= {len(data):,d} samples)')
    ax.set_ylabel('count measurments')
    _ = ax.legend()
    vaep.savefig(
        fig, folder / f'hist_{gene_selected}_pg_{pg_selected.split(";")[0]}.pdf')
    plt.close(fig)
# %% [markdown]
# ## Compare with target annotation

# %%
# labels somehow?
# target.replace({True: f' >={args.cutoff_target}', False: f'<{args.cutoff_target}'})

for idx in differences.index:
    pg_selected, gene_selected = idx  # top feat

    _series, _series_vae, _series_rsn = (
        data[pg_selected].dropna(),
        pred_real_na[pg_selected].dropna(),
        pred_real_na_imputed_baseline[pg_selected].dropna()
    )
    ax = None
    groups_order = [f'Measured (N={len(_series):,d})',
                    f'{args.model_key.upper()} (N={len(_series_vae):,d}, q={differences.loc[idx, ("VAE", "qvalue")]:.3f})',
                    f'RSN (N={len(_series_rsn):,d}, q={differences.loc[idx, ("RSN", "qvalue")]:.3f})']
    to_plot = pd.concat([
        _series.to_frame('intensity').assign(
            group=groups_order[0]),
        _series_vae.to_frame('intensity').assign(
            group=groups_order[1]),
        _series_rsn.to_frame('intensity').assign(
            group=groups_order[2]),
    ]).join(target, how='inner')

    ax = seaborn.swarmplot(data=to_plot,
                           x='group',
                           y='intensity',
                           order=groups_order,
                             hue=args.target)
    fig = ax.get_figure()
    ax.set_title(
        f'Imputation for protein group {pg_selected.split(";")[0]} (gene: {gene_selected.split(";")[0]}) with target {target_name} (N= {len(data):,d} samples)')

    _ = ax.legend()
    _ = ax.set_ylim(min_y_int, max_y_int)
    _ = ax.locator_params(axis='y', integer=True)
    _ = ax.set_xlabel('')
    fname = (folder /
             f'swarmplot_{gene_selected.split(";")[0]}'
             f'_pg_{pg_selected.split(";")[0]}.pdf')
    vaep.savefig(
        fig,
        name=fname)
    plt.close()


# %% [markdown]
# - add non-imputed data q-value
# %%
