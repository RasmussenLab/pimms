# %% [markdown]
# # Plots for comparison on ALD dataset with 20% add MAR values

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import njab
import pandas as pd

import vaep

plt.rcParams['figure.figsize'] = [4, 2]  # [16.0, 7.0] , [4, 3]
vaep.plotting.make_large_descriptors(6)


NONE_COL_NAME = 'No imputation\n(None)'
col_mapper = {'None':
              NONE_COL_NAME}
# overwrite for now to align with Fig. 3
ORDER_MODELS = ['DAE', 'VAE', 'TRKNN', 'RF', 'CF', 'Median', 'QRILC', NONE_COL_NAME]
REF_MODEL = 'None (100%)'
CUTOFF = 0.05

COLORS_TO_USE_MAPPTING = vaep.plotting.defaults.color_model_mapping
COLORS_TO_USE_MAPPTING[NONE_COL_NAME] = COLORS_TO_USE_MAPPTING['None']

COLORS_CONTIGENCY_TABLE = {
    k: f'C{i}' for i, k in enumerate(['FP', 'TN', 'TP', 'FN'])
}


def plot_qvalues(df, x: str, y: list, ax=None, cutoff=0.05,
                 alpha=1.0, style='.', markersize=3):
    ax = df.plot.line(x=x,
                      y=y,
                      style=style,
                      ax=ax,
                      color=COLORS_TO_USE_MAPPTING,
                      alpha=alpha,
                      markersize=markersize)
    _ = ax.hlines(cutoff,
                  xmin=ax.get_xlim()[0],
                  xmax=ax.get_xlim()[1],
                  linestyles='dashed',
                  color='grey',
                  linewidth=1)
    return ax


# %% [markdown]
# DA analysis
# %%
out_folder = 'runs/appl_ald_data_2023_11/plasma/proteinGroups_80perc_25MNAR/diff_analysis/kleiner/'
out_folder = Path(out_folder)

# %%
files_out = dict()
fname = out_folder / 'ald_reduced_dataset_plots.xlsx'
files_out[fname.name] = fname.as_posix()
writer = pd.ExcelWriter(fname)


# %%


# %% [markdown]
# Load dumps

# %%
da_target = (pd
             .read_pickle(out_folder / 'equality_rejected_target.pkl').
             rename(col_mapper, axis=1)
             )
da_target.describe()

# %%
qvalues = (pd
           .read_pickle(out_folder / 'qvalues_target.pkl')
           .rename(col_mapper, axis=1)
           )
qvalues

# %% [markdown]
# take only those with different decisions

# %%
da_target = da_target.drop('RSN', axis=1)
da_target_same = (da_target.sum(axis=1) == 0) | da_target.all(axis=1)
da_target_same.value_counts()


# %%
feat_idx_w_diff = da_target_same[~da_target_same].index
feat_idx_w_diff

# %%
qvalues_sel = (qvalues
               .loc[feat_idx_w_diff]
               .sort_values((NONE_COL_NAME, 'qvalue')
                            ))


# %%
da_target_sel = da_target.loc[qvalues_sel.index]
da_target_sel

# %% [markdown]
# ## Diff. abundant => not diff. abundant

# %%
mask_lost_sign = (
    (da_target_sel[NONE_COL_NAME] == False)
    & (da_target_sel[REF_MODEL])
)
sel = qvalues_sel.loc[mask_lost_sign.squeeze()]
sel.columns = sel.columns.droplevel(-1)
sel = sel[ORDER_MODELS + [REF_MODEL]].sort_values(REF_MODEL)
sel.to_excel(writer, sheet_name='lost_signal_qvalues')
sel

# %%
# 0: FN
# 1: TP
da_target_sel_counts = (da_target_sel[ORDER_MODELS]
                        .loc[mask_lost_sign.squeeze()]
                        .astype(int)
                        .replace(
    {0: 'FN',
     1: 'TP'}
).droplevel(-1, axis=1)
)
da_target_sel_counts = njab.pandas.combine_value_counts(da_target_sel_counts)
ax = da_target_sel_counts.T.plot.bar(ylabel='count',
                                     color=[COLORS_CONTIGENCY_TABLE['FN'],
                                            COLORS_CONTIGENCY_TABLE['TP']])
ax.locator_params(axis='y', integer=True)
fname = out_folder / 'lost_signal_da_counts.pdf'
da_target_sel_counts.fillna(0).to_excel(writer, sheet_name=fname.stem)
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %%
ax = plot_qvalues(df=sel,
                  x=REF_MODEL,
                  y=ORDER_MODELS,
                  cutoff=CUTOFF)
ax.set_xlim(-0.0005, CUTOFF + 0.015)
ax.legend(loc='upper right')
ax.set_xlabel("q-value using 100% of the data without imputation")
ax.set_ylabel("q-value using 80% of the data")
fname = out_folder / 'lost_signal_qvalues.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)


# %% [markdown]
# ## Not diff. abundant => diff. abundant

# %%
mask_gained_signal = (
    (da_target_sel[NONE_COL_NAME])
    & (da_target_sel[REF_MODEL] == False)
)
sel = qvalues_sel.loc[mask_gained_signal.squeeze()]
sel.columns = sel.columns.droplevel(-1)
sel = sel[ORDER_MODELS + [REF_MODEL]].sort_values(REF_MODEL)
sel.to_excel(writer, sheet_name='gained_signal_qvalues')
sel

# %%
da_target_sel_counts = (da_target_sel[ORDER_MODELS]
                        .loc[mask_gained_signal.squeeze()]
                        .astype(int)
                        .replace(
    {0: 'TN',
     1: 'FP'}
).droplevel(-1, axis=1)
)
da_target_sel_counts = njab.pandas.combine_value_counts(da_target_sel_counts)
ax = da_target_sel_counts.T.plot.bar(ylabel='count',
                                     color=[COLORS_CONTIGENCY_TABLE['TN'],
                                            COLORS_CONTIGENCY_TABLE['FP']])
ax.locator_params(axis='y', integer=True)
fname = out_folder / 'gained_signal_da_counts.pdf'
da_target_sel_counts.fillna(0).to_excel(writer, sheet_name=fname.stem)
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %%
ax = plot_qvalues(sel,
                  x=REF_MODEL,
                  y=ORDER_MODELS)
# ax.set_xlim(CUTOFF - 0.005, sel[REF_MODEL].max() + 0.005)
ax.set_xlabel("q-value using 100% of the data without imputation")
ax.set_ylabel("q-value using 80% of the data")
ax.legend(loc='upper right')
fname = out_folder / 'gained_signal_qvalues.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %% [markdown]
# Saved files

# %%
writer.close()
files_out

# %%
