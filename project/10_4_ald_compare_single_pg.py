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

import matplotlib
import seaborn

import vaep
import vaep.analyzers
import vaep.io.datasplits
import vaep.imputation

logger = vaep.logging.setup_nb_logger()


plt.rcParams['figure.figsize'] = [4, 2.5]  # [16.0, 7.0] , [4, 3]
vaep.plotting.make_large_descriptors(5)

# %% [markdown]
# ## Parameters

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
folder_experiment = 'runs/appl_ald_data/plasma/proteinGroups'
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"
make_plots = True # create histograms and swarmplots of diverging results
model_key = 'VAE'
sample_id_col = 'Sample ID'
target = 'kleiner'
cutoff_target: int = 2  # => for binarization target >= cutoff_target
out_folder = 'diff_analysis'
file_format = 'csv'
baseline = 'RSN'  # default is RSN, but could be any other trained model
template_pred = 'pred_real_na_{}.csv'  # fixed, do not change


# %%
params = vaep.nb.get_params(args, globals=globals())
params

# %%
args = vaep.nb.Config()
args.folder_experiment = Path(params["folder_experiment"])
args = vaep.nb.add_default_paths(args,
                                 out_root=(args.folder_experiment
                                           / params["out_folder"]
                                           / params["target"]))
args.folder_scores = (args.folder_experiment
                      / params["out_folder"]
                      / params["target"]
                      / 'scores'
                      )
args.update_from_dict(params)
args

# %%
files_out = dict()

fname = args.out_folder / 'diff_analysis_compare_DA.xlsx'
writer = pd.ExcelWriter(fname)
files_out[fname.name] = fname.as_posix()


# %%
score_dumps = [fname for fname in Path(
    args.folder_scores).iterdir() if fname.suffix == '.pkl']
score_dumps

# %%
scores = pd.concat([pd.read_pickle(fname) for fname in score_dumps], axis=1)
scores

# %%
# %% [markdown]
# ## Load frequencies of observed features

# %%
fname = args.folder_experiment / 'freq_features_observed.csv'
freq_feat = pd.read_csv(fname, index_col=0)
freq_feat.columns = pd.MultiIndex.from_tuples([('data', 'frequency'),])
freq_feat

# %%
qvalues = scores.loc[pd.IndexSlice[:, args.target],
                     pd.IndexSlice[:, 'qvalue']
                     ].join(freq_feat
                            ).set_index(
    ('data', 'frequency'), append=True)
qvalues.index.names = qvalues.index.names[:-1] + ['frequency']
fname  = args.out_folder / 'qvalues_target.pkl'
files_out[fname.name] = fname.as_posix()
qvalues.to_pickle(fname)
qvalues.to_excel(writer, sheet_name='qvalues_all')
qvalues

# %%
pvalues = scores.loc[pd.IndexSlice[:, args.target],
                     pd.IndexSlice[:, 'p-unc']
                     ].join(freq_feat
                            ).set_index(
    ('data', 'frequency'), append=True)
pvalues.index.names = pvalues.index.names[:-1] + ['frequency']
fname  = args.out_folder / 'pvalues_target.pkl'
files_out[fname.name] = fname.as_posix()
pvalues.to_pickle(fname)
pvalues.to_excel(writer, sheet_name='pvalues_all')
pvalues

# %%
da_target = scores.loc[pd.IndexSlice[:, args.target],
                       pd.IndexSlice[:, 'rejected']
                       ].join(freq_feat
                            ).set_index(
    ('data', 'frequency'), append=True)
da_target.index.names = da_target.index.names[:-1] + ['frequency']
fname = args.out_folder / 'equality_rejected_target.pkl'
files_out[fname.name] = fname.as_posix()
da_target.to_pickle(fname)

count_rejected = vaep.pandas.combine_value_counts(da_target.droplevel(-1, axis=1))
count_rejected.to_excel(writer, sheet_name='count_rejected')
count_rejected

# %%
mask_common = da_target.notna().all(axis=1)
count_rejected_common = vaep.pandas.combine_value_counts(da_target.loc[mask_common].droplevel(-1, axis=1))
count_rejected_common.to_excel(writer, sheet_name='count_rejected_common')
count_rejected_common

# %%
count_rejected_new = vaep.pandas.combine_value_counts(da_target.loc[~mask_common].droplevel(-1, axis=1))
count_rejected_new.to_excel(writer, sheet_name='count_rejected_new')
count_rejected_new


# %%
da_target.to_excel(writer, sheet_name='equality_rejected_all')
da_target

# %%
da_target_same = (da_target.sum(axis=1) == 0) | da_target.all(axis=1)
da_target_same.value_counts()

# %%
feat_idx_w_diff = da_target_same[~da_target_same].index
feat_idx_w_diff

# %% [markdown]
# take only those with different decisions

# %%
(qvalues
 .loc[feat_idx_w_diff]
 .sort_values(('None', 'qvalue'))
 .to_excel(writer, sheet_name='qvalues_diff'))
writer.close()

# %% [markdown]
# ## Plots for inspecting imputations (for diverging decisions)

# %%
if not args.make_plots:
    logger.warning("Not plots requested.")
    import sys
    sys.exit(0)


# %% [markdown]
# ## Load target

# %%
target = pd.read_csv(args.fn_clinical_data,
                     index_col=0,
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
    args.data,
    file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y]).unstack()
data

# %% [markdown]
# plot all of the new pgs which are at least once significant which are not already dumped.

# %%
feat_new_abundant = da_target.loc[~mask_common].any(axis=1)
feat_new_abundant = feat_new_abundant.loc[feat_new_abundant].index.get_level_values(0)
feat_new_abundant

# %%
feat_sel = feat_idx_w_diff.get_level_values(0)
feat_sel = feat_sel.union(feat_new_abundant)
len(feat_sel)

# %%
data = data.loc[:, feat_sel]
data

# %% [markdown]
# - RSN prediction are based on all samples mean and std (N=455) as in original study
# - VAE also trained on all samples (self supervised)
# One could also reduce the selected data to only the samples with a valid target marker,
# but this was not done in the original study which considered several different target markers.
#
# RSN : shifted per sample, not per feature!
#
# Load all prediction files and reshape

# %%
# exclude 'None' as this is without imputation (-> data)
model_keys = [k for k in qvalues.columns.get_level_values(0) if k != 'None']
pred_paths = [
    args.out_preds / args.template_pred.format(method)
    for method in model_keys]
pred_paths

# %%
load_single_csv_pred_file = vaep.analyzers.compare_predictions.load_single_csv_pred_file
pred_real_na = dict()
for method in model_keys:
    fname = args.out_preds / args.template_pred.format(method)
    print(f"missing values pred. by {method}: {fname}")
    pred_real_na[method] = load_single_csv_pred_file(fname)
pred_real_na = pd.DataFrame(pred_real_na)
pred_real_na


# %% [markdown]
# Once imputation, reduce to target samples only (samples with target score)

# %%
# select samples with target information
data = data.loc[target.index]
pred_real_na = pred_real_na.loc[target.index]

# assert len(data) == len(pred_real_na)


# %%
idx = feat_sel[0]

# %%
feat_observed = data[idx].dropna()
feat_observed

# %%
# axes = axes.ravel()
# args.out_folder.parent / 'intensity_plots'
# each feature -> one plot?
# plot all which are at least for one method significant?
folder = args.out_folder / 'intensities_for_diff_in_DA_decision'
folder.mkdir(parents=True, exist_ok=True)


# %%
min_y_int, max_y_int = vaep.plotting.data.get_min_max_iterable(
    [data.stack(), pred_real_na.stack()])
min_max = min_y_int, max_y_int

target_name = target.columns[0]

min_max, target_name

# %%
for idx in feat_sel:
    fig, ax = plt.subplots()

    feat_observed = data[idx].dropna()

    label_template = '{method} (N={n:,d}, q={q:.3f})'
    # observed data
    vaep.plotting.data.plot_histogram_intensites(
        feat_observed,
        ax=ax,
        min_max=min_max,
        label=label_template.format(method='measured',
                                    n=len(feat_observed),
                                    q=float(qvalues.loc[idx, ('None', 'qvalue')])),
        color='grey',
        alpha=0.6)

    # all models
    for i, method in enumerate(model_keys):
        try:
            pred = pred_real_na.loc[pd.IndexSlice[:, idx], method].dropna()
            if len(pred) == 0:
                # in case no values was imputed -> qvalue is as based on measured
                label = label_template.format(method=method,
                                              n=len(pred),
                                              q=float(qvalues.loc[idx, ('None', 'qvalue')]
                                                      ))
            else:
                label = label_template.format(method=method,
                                              n=len(pred),
                                              q=float(qvalues.loc[idx, (method, 'qvalue')]
                                                      ))
            ax, bins = vaep.plotting.data.plot_histogram_intensites(
                pred,
                ax=ax,
                min_max=min_max,
                label=label,
                color=f'C{i}',
                alpha=0.6)
        except KeyError:
            print(f"No missing values for {idx}: {method}")
            continue
    first_pg = idx.split(";")[0]
    ax.set_title(
        f'Imputation for protein group {first_pg} with target {target_name} (N= {len(data):,d} samples)')
    ax.set_ylabel('count measurments')
    _ = ax.legend()
    files_out[fname.name] = fname.as_posix()
    vaep.savefig(
        fig, folder / f'{first_pg}_hist.pdf')
    plt.close(fig)

# %% [markdown]
# ## Compare with target annotation

# %%
# labels somehow?
# target.replace({True: f' >={args.cutoff_target}', False: f'<{args.cutoff_target}'})

for i, idx in enumerate(feat_sel):
    print(f"Swarmplot {i:3<}: {idx}:")
    fig, ax = plt.subplots()

    feat_observed = data[idx].dropna()
    label_template = '{method} (N={n:,d}, q={q:.3f})'
    key = label_template.format(method='measured',
                                n=len(feat_observed),
                                q=float(qvalues.loc[idx, ('None', 'qvalue')])
                                )
    to_plot = {key: feat_observed}

    for method in model_keys:
        try:
            pred = pred_real_na.loc[pd.IndexSlice[:,
                                                  idx], method].dropna().droplevel(-1)
            if len(pred) == 0:
                # in case no values was imputed -> qvalue is as based on measured
                key = label_template.format(method=method,
                                            n=len(pred),
                                            q=float(qvalues.loc[idx, ('None', 'qvalue')]
                                                    ))
            elif qvalues.loc[idx, (method, 'qvalue')].notna().all():
                key = label_template.format(method=method,
                                            n=len(pred),
                                            q=float(qvalues.loc[idx, (method, 'qvalue')]
                                                    ))
            elif qvalues.loc[idx, (method, 'qvalue')].isna().all():
                logger.info(f"NA qvalues for {idx}: {method}")
                continue
            else:
                raise ValueError("Unknown case.")
            to_plot[key] = pred
        except KeyError:
            print(f"No missing values for {idx}: {method}")
            continue

    to_plot = pd.DataFrame.from_dict(to_plot)
    to_plot.columns.name = 'group'
    groups_order = to_plot.columns.to_list()
    to_plot = to_plot.stack().to_frame('intensity').reset_index(-1)
    to_plot = to_plot.join(target.astype('category'), how='inner')

    ax = seaborn.swarmplot(data=to_plot,
                           x='group',
                           y='intensity',
                           order=groups_order,
                           hue=args.target,
                           size=2,
                           ax=ax)
    first_pg = idx.split(";")[0]
    ax.set_title(
        f'Imputation for protein group {first_pg} with target {target_name} (N= {len(data):,d} samples)')

    _ = ax.legend(fontsize=5, title_fontsize=5, markerscale=0.4,)
    _ = ax.set_ylim(min_y_int, max_y_int)
    _ = ax.locator_params(axis='y', integer=True)
    _ = ax.set_xlabel('')
    _xticks = ax.get_xticks()
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator(_xticks)
    )
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           horizontalalignment='right')
    fname = (folder /
             f'{first_pg}_swarmplot.pdf')
    files_out[fname.name] = fname.as_posix()
    vaep.savefig(
        fig,
        name=fname)
    plt.close()


# %%
files_out
