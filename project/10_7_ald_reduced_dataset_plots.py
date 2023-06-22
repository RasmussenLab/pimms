# %% [markdown]
# # Plots for comparison on ALD dataset with 20% add MAR values

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import vaep
plt.rcParams['figure.figsize'] = [4, 2]  # [16.0, 7.0] , [4, 3]
vaep.plotting.make_large_descriptors(5)

COLORS_TO_USE_MAPPTING = vaep.plotting.defaults.color_model_mapping

def plot_qvalues(df, x: str, y: list, ax=None, cutoff=0.05,
                 alpha=1.0):
    ax = df.plot.line(x=x,
                      y=y,
                      style='.',
                      ax=ax,
                      color=COLORS_TO_USE_MAPPTING,
                      alpha=alpha,
                      markersize=1.5)
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
out_folder = 'runs/appl_ald_data/plasma/proteinGroups_80%_dataset/diff_analysis/kleiner/'
out_folder = Path(out_folder)

# %%
files_out = dict()
fname = out_folder / 'ald_reduced_dataset_plots.xlsx'
files_out[fname.name] = fname.as_posix()
writer = pd.ExcelWriter(fname)

# %% [markdown]
# Ordering of model and reference model

# %%
ORDER_MODELS = pd.read_csv(
    out_folder.parent.parent / 'figures/performance_test.csv',
    index_col=0
).index.to_list()
ORDER_MODELS

# %%
# overwrite for now to align with Fig. 3
ORDER_MODELS = ['DAE', 'VAE', 'rf', 'CF', 'KNN', 'Median', 'None']
REF_MODEL = 'None (100%)'
CUTOFF = 0.05

# %% [markdown]
# Load dumps

# %%
da_target = pd.read_pickle(out_folder / 'equality_rejected_target.pkl')
da_target.describe()

# %%
qvalues = pd.read_pickle(out_folder / 'qvalues_target.pkl')
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
               .sort_values(('None', 'qvalue')
                            ))


# %%
da_target_sel = da_target.loc[qvalues_sel.index]
da_target_sel

# %% [markdown]
# ## Diff. abundant => not diff. abundant

# %%
mask_lost_sign = (
    (da_target_sel['None'] == False)
    & (da_target_sel[REF_MODEL] == True)
)
sel = qvalues_sel.loc[mask_lost_sign.squeeze()]
sel.columns = sel.columns.droplevel(-1)
sel = sel[ORDER_MODELS + [REF_MODEL]]
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
da_target_sel_counts = vaep.pandas.combine_value_counts(da_target_sel_counts)
ax = da_target_sel_counts.T.plot.bar()
ax.locator_params(axis='y', integer=True)
fname = out_folder / 'lost_signal_da_counts.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %%
ax = plot_qvalues(df=sel,
                  x=REF_MODEL,
                  y=ORDER_MODELS,
                  cutoff=CUTOFF)
ax.set_xlim(-0.0005, CUTOFF + 0.0005)
ax.set_xlabel("q-value using 100% of the data without imputation")
ax.set_ylabel("q-value using 80%")
fname = out_folder / 'lost_signal_qvalues.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)


# %% [markdown]
# ## Not diff. abundant => diff. abundant

# %%
mask_gained_signal = (
    (da_target_sel['None'] == True)
    & (da_target_sel[REF_MODEL] == False)
)
sel = qvalues_sel.loc[mask_gained_signal.squeeze()]
sel.columns = sel.columns.droplevel(-1)
sel = sel[ORDER_MODELS + [REF_MODEL]]
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
da_target_sel_counts = vaep.pandas.combine_value_counts(da_target_sel_counts)
ax = da_target_sel_counts.T.plot.bar()
ax.locator_params(axis='y', integer=True)
fname = out_folder / 'gained_signal_da_counts.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %%
ax = plot_qvalues(sel,
                  x=REF_MODEL,
                  y=ORDER_MODELS)
ax.set_xlim(CUTOFF - 0.01, sel[REF_MODEL].max() + 0.005)
ax.set_xlabel("q-value using 100% of the data without imputation")
ax.set_ylabel("q-value using 80%")
ax.legend(loc='upper center')
fname = out_folder / 'gained_signal_qvalues.pdf'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.figure, fname)

# %% [markdown]
# Saved files

# %%
writer.close()
files_out

# %%
