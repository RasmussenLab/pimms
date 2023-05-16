# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %%
from pathlib import Path
import pandas as pd
import vaep

# %%
pickled_qvalues = snakemake.input.qvalues
pickled_qvalues

# %%
files_out = dict()
folder_out = Path(snakemake.params.folder_experiment)
fname = folder_out / 'agg_differences_compared.xlsx'
writer = pd.ExcelWriter(fname)
fname

# %%
def _load_pickle(pfath, run:int):
    df = pd.read_pickle(pfath)
    df['run'] = f'run{run:02d}'
    df = df.set_index('run', append=True)
    return df

df_long_qvalues = pd.concat(
    [_load_pickle(f,i) for i,f  in enumerate(pickled_qvalues)]
    )
df_long_qvalues

# %% [markdown]
# Q-values for features across runs

# %%
qvalue_stats = df_long_qvalues.groupby(level=0).agg(['mean', 'std'])
qvalue_stats.to_excel(writer, sheet_name='all_qvalue_stats')
qvalue_stats

# %%
decisions_da_target = snakemake.input.equality_rejected_target
decisions_da_target

# %%
da_counts = sum(pd.read_pickle(f) for f in decisions_da_target)
da_counts.to_excel(writer, sheet_name='all_rejected_counts')
da_counts

# %% [markdown]
# Option: set custom qvalue threshold

# %%
qvalue_treshold = 0.05
da_counts = sum(pd.read_pickle(f) < qvalue_treshold for f in pickled_qvalues)
da_counts

# %%
da_target_same = (da_counts.sum(axis=1) == 0) | da_counts.all(axis=1)
da_target_same.value_counts()

# %%
idx_different = (da_target_same
                 [~da_target_same]
                 .index
                 .get_level_values(0)
)

# %%
da_counts = da_counts.loc[idx_different]
da_counts

# %% [markdown]
# Order by mean qvalue of non-imputed comparison

# %%
qvalue_stats = (qvalue_stats
                .loc[idx_different]
                .sort_values(('None', 'qvalue', 'mean'))
)
# qvalue_stats.to_excel(writer, sheet_name='different_qvalue_stats')
qvalue_stats

# %% [markdown]
# save more verbose index on scores, transfer to counts

# %%
da_counts = da_counts.loc[qvalue_stats.index]
da_counts.to_excel(writer, sheet_name='different_rejected_counts')
qvalue_stats.index = da_counts.index
qvalue_stats.to_excel(writer, sheet_name='different_qvalue_stats')

# %%
fname
