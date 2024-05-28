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
# # Compare predictions between model and RSN
#
# - see differences in imputation for diverging cases
# - dumps top5

# %% tags=["hide-input"]
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import njab
import pandas as pd
import seaborn

import vaep
import vaep.analyzers
import vaep.imputation
import vaep.io.datasplits

logger = vaep.logging.setup_nb_logger()
logging.getLogger('fontTools').setLevel(logging.WARNING)

plt.rcParams['figure.figsize'] = [4, 2.5]  # [16.0, 7.0] , [4, 3]
vaep.plotting.make_large_descriptors(7)

# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
folder_experiment = 'runs/appl_ald_data/plasma/proteinGroups'
fn_clinical_data = "data/ALD_study/processed/ald_metadata_cli.csv"
make_plots = True  # create histograms and swarmplots of diverging results
model_key = 'VAE'
sample_id_col = 'Sample ID'
target = 'kleiner'
cutoff_target: int = 2  # => for binarization target >= cutoff_target
out_folder = 'diff_analysis'
file_format = 'csv'
baseline = 'RSN'  # default is RSN, but could be any other trained model
template_pred = 'pred_real_na_{}.csv'  # fixed, do not change
ref_method_score = None  # filepath to reference method score


# %% tags=["hide-input"]
params = vaep.nb.get_params(args, globals=globals())
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

# %% [markdown]
# Write outputs to excel

# %% tags=["hide-input"]
files_out = dict()

fname = args.out_folder / 'diff_analysis_compare_DA.xlsx'
writer = pd.ExcelWriter(fname)
files_out[fname.name] = fname.as_posix()
logger.info("Writing to excel file: %s", fname)

# %% [markdown]
# ## Load scores
# List dump of scores:

# %% tags=["hide-input"]
score_dumps = [fname for fname in Path(
    args.folder_scores).iterdir() if fname.suffix == '.pkl']
score_dumps

# %% [markdown]
# Load scores from dumps:

# %% tags=["hide-input"]
scores = pd.concat([pd.read_pickle(fname) for fname in score_dumps], axis=1)
scores

# %% [markdown]
# If reference dump is provided, add it to the scores

# %% tags=["hide-input"]
if args.ref_method_score:
    scores_reference = (pd
                        .read_pickle(args.ref_method_score)
                        .rename({'None': 'None (100%)'},
                                axis=1))
    scores = scores.join(scores_reference)
    logger.info(f'Added reference method scores from {args.ref_method_score}')

# %% [markdown]
# ### Load frequencies of observed features

# %% tags=["hide-input"]
fname = args.folder_experiment / 'freq_features_observed.csv'
freq_feat = pd.read_csv(fname, index_col=0)
freq_feat.columns = pd.MultiIndex.from_tuples([('data', 'frequency'),])
freq_feat

# %% [markdown]
# ### Assemble qvalues

# %% tags=["hide-input"]
qvalues = scores.loc[pd.IndexSlice[:, args.target],
                     pd.IndexSlice[:, 'qvalue']
                     ].join(freq_feat
                            ).set_index(
    ('data', 'frequency'), append=True)
qvalues.index.names = qvalues.index.names[:-1] + ['frequency']
fname = args.out_folder / 'qvalues_target.pkl'
files_out[fname.name] = fname.as_posix()
qvalues.to_pickle(fname)
qvalues.to_excel(writer, sheet_name='qvalues_all')
qvalues

# %% [markdown]
# ### Assemble pvalues

# %% tags=["hide-input"]
pvalues = scores.loc[pd.IndexSlice[:, args.target],
                     pd.IndexSlice[:, 'p-unc']
                     ].join(freq_feat
                            ).set_index(
    ('data', 'frequency'), append=True)
pvalues.index.names = pvalues.index.names[:-1] + ['frequency']
fname = args.out_folder / 'pvalues_target.pkl'
files_out[fname.name] = fname.as_posix()
pvalues.to_pickle(fname)
pvalues.to_excel(writer, sheet_name='pvalues_all')
pvalues

# %% [markdown]
# ### Assemble rejected features

# %% tags=["hide-input"]
da_target = scores.loc[pd.IndexSlice[:, args.target],
                       pd.IndexSlice[:, 'rejected']
                       ].join(freq_feat
                              ).set_index(
    ('data', 'frequency'), append=True)
da_target.index.names = da_target.index.names[:-1] + ['frequency']
fname = args.out_folder / 'equality_rejected_target.pkl'
files_out[fname.name] = fname.as_posix()
da_target.to_pickle(fname)
count_rejected = njab.pandas.combine_value_counts(da_target.droplevel(-1, axis=1))
count_rejected.to_excel(writer, sheet_name='count_rejected')
count_rejected

# %% [markdown]
# ### Tabulate rejected decisions by method:

# %% tags=["hide-input"]
# # ! This uses implicitly that RSN is not available for some protein groups
# # ! Make an explicit list of the 313 protein groups available in original data
mask_common = da_target.notna().all(axis=1)
count_rejected_common = njab.pandas.combine_value_counts(da_target.loc[mask_common].droplevel(-1, axis=1))
count_rejected_common.to_excel(writer, sheet_name='count_rejected_common')
count_rejected_common

# %% [markdown]
# ### Tabulate rejected decisions by method for newly included features (if available)

# %% tags=["hide-input"]
count_rejected_new = njab.pandas.combine_value_counts(da_target.loc[~mask_common].droplevel(-1, axis=1))
count_rejected_new.to_excel(writer, sheet_name='count_rejected_new')
count_rejected_new

# %% [markdown]
# ### Tabulate rejected decisions by method for all features

# %% tags=["hide-input"]
da_target.to_excel(writer, sheet_name='equality_rejected_all')
logger.info("Written to sheet 'equality_rejected_all' in excel file.")
da_target

# %% [markdown]
# Tabulate number of equal decison by method (`True`) to the ones with varying 
# decision depending on the method (`False`)

# %% tags=["hide-input"]
da_target_same = (da_target.sum(axis=1) == 0) | da_target.all(axis=1)
da_target_same.value_counts()

# %% [markdown]
# List frequency of features with varying decisions

# %% tags=["hide-input"]
feat_idx_w_diff = da_target_same[~da_target_same].index
feat_idx_w_diff.to_frame()[['frequency']].reset_index(-1, drop=True)

# %% [markdown]
# take only those with different decisions

# %% tags=["hide-input"]
(qvalues
 .loc[feat_idx_w_diff]
 .sort_values(('None', 'qvalue'))
 .to_excel(writer, sheet_name='qvalues_diff')
 )

(qvalues
 .loc[feat_idx_w_diff]
 .loc[mask_common]  # mask automatically aligned
 .sort_values(('None', 'qvalue'))
 .to_excel(writer, sheet_name='qvalues_diff_common')
 )

try:
    (qvalues
     .loc[feat_idx_w_diff]
     .loc[~mask_common]
     .sort_values(('None', 'qvalue'))
     .to_excel(writer, sheet_name='qvalues_diff_new')
     )
except IndexError:
    print("No new features or no new ones (with diverging decisions.)")
writer.close()

# %% [markdown]
# ## Plots for inspecting imputations (for diverging decisions)

# %% tags=["hide-input"]
if not args.make_plots:
    logger.warning("Not plots requested.")
    import sys
    sys.exit(0)


# %% [markdown]
# ## Load target

# %% tags=["hide-input"]
target = pd.read_csv(args.fn_clinical_data,
                     index_col=0,
                     usecols=[args.sample_id_col, args.target])
target = target.dropna()
target

# %% tags=["hide-input"]
target_to_group = target.copy()
target = target >= args.cutoff_target
target = target.replace({False: f'{args.target} < {args.cutoff_target}',
                        True: f'{args.target} >= {args.cutoff_target}'}
                        ).astype('category')
pd.crosstab(target.squeeze(), target_to_group.squeeze())

# %% [markdown]
# ## Measurments

# %% tags=["hide-input"]
data = vaep.io.datasplits.DataSplits.from_folder(
    args.data,
    file_format=args.file_format)
data = pd.concat([data.train_X, data.val_y, data.test_y]).unstack()
data

# %% [markdown]
# plot all of the new pgs which are at least once significant which are not already dumped.

# %% tags=["hide-input"]
feat_new_abundant = da_target.loc[~mask_common].any(axis=1)
feat_new_abundant = feat_new_abundant.loc[feat_new_abundant].index.get_level_values(0)
feat_new_abundant

# %% tags=["hide-input"]
feat_sel = feat_idx_w_diff.get_level_values(0)
feat_sel = feat_sel.union(feat_new_abundant)
len(feat_sel)

# %% tags=["hide-input"]
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

# %% tags=["hide-input"]
# exclude 'None' as this is without imputation (-> data)
model_keys = [k for k in qvalues.columns.get_level_values(0) if k != 'None']
pred_paths = [
    args.out_preds / args.template_pred.format(method)
    for method in model_keys]
pred_paths

# %% tags=["hide-input"]
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

# %% tags=["hide-input"]
# select samples with target information
data = data.loc[target.index]
pred_real_na = pred_real_na.loc[target.index]

# assert len(data) == len(pred_real_na)


# %% tags=["hide-input"]
idx = feat_sel[0]

# %% tags=["hide-input"]
feat_observed = data[idx].dropna()
feat_observed

# %% tags=["hide-input"]
# axes = axes.ravel()
# args.out_folder.parent / 'intensity_plots'
# each feature -> one plot?
# plot all which are at least for one method significant?
folder = args.out_folder / 'intensities_for_diff_in_DA_decision'
folder.mkdir(parents=True, exist_ok=True)


# %% tags=["hide-input"]
min_y_int, max_y_int = vaep.plotting.data.get_min_max_iterable(
    [data.stack(), pred_real_na.stack()])
min_max = min_y_int, max_y_int

target_name = target.columns[0]

min_max, target_name


# %% [markdown]
# ## Compare with target annotation

# %% tags=["hide-input"]
# labels somehow?
# target.replace({True: f' >={args.cutoff_target}', False: f'<{args.cutoff_target}'})

for i, idx in enumerate(feat_sel):
    print(f"Swarmplot {i:3<}: {idx}:")
    fig, ax = plt.subplots()

    # dummy plots, just to get the Path objects
    tmp_dot = ax.scatter([1, 2], [3, 4], marker='X')
    new_mk, = tmp_dot.get_paths()
    tmp_dot.remove()

    feat_observed = data[idx].dropna()

    def get_centered_label(method, n, q):
        model_str = f'{method}'
        stats_str = f'(N={n:,d}, q={q:.3f})'
        if len(model_str) > len(stats_str):
            stats_str = f"{stats_str:<{len(model_str)}}"
        else:
            model_str = f"{model_str:<{len(stats_str)}}"
        return f'{model_str}\n{stats_str}'

    key = get_centered_label(method='observed',
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
                key = get_centered_label(method=method,
                                         n=len(pred),
                                         q=float(qvalues.loc[idx, ('None', 'qvalue')]
                                                 ))
            elif qvalues.loc[idx, (method, 'qvalue')].notna().all():
                key = get_centered_label(method=method,
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
    to_plot = to_plot.astype({'group': 'category'})

    ax = seaborn.swarmplot(data=to_plot,
                           x='group',
                           y='intensity',
                           order=groups_order,
                           dodge=True,
                           hue=args.target,
                           size=2,
                           ax=ax)
    first_pg = idx.split(";")[0]
    ax.set_title(
        f'Imputation for protein group {first_pg} with target {target_name} (N= {len(data):,d} samples)')

    _ = ax.set_ylim(min_y_int, max_y_int)
    _ = ax.locator_params(axis='y', integer=True)
    _ = ax.set_xlabel('')
    _xticks = ax.get_xticks()
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator(_xticks)
    )
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           horizontalalignment='right')

    N_hues = len(pd.unique(to_plot[args.target]))

    _ = ax.collections[0].set_paths([new_mk])
    _ = ax.collections[1].set_paths([new_mk])

    label_target_0, label_target_1 = ax.collections[-2].get_label(), ax.collections[-1].get_label()
    _ = ax.collections[-2].set_label(f'imputed, {label_target_0}')
    _ = ax.collections[-1].set_label(f'imputed, {label_target_1}')
    _obs_label0 = ax.scatter([], [], color='C0', marker='X', label=f'observed, {label_target_0}')
    _obs_label1 = ax.scatter([], [], color='C1', marker='X', label=f'observed, {label_target_1}')
    _ = ax.legend(
        handles=[_obs_label0, _obs_label1, *ax.collections[-4:-2]],
        fontsize=5, title_fontsize=5, markerscale=0.4,)
    fname = (folder /
             f'{first_pg}_swarmplot.pdf')
    files_out[fname.name] = fname.as_posix()
    vaep.savefig(
        fig,
        name=fname)
    plt.close()

# %% [markdown]
# Saved files:

# %% tags=["hide-input"]
files_out
