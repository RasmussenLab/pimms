# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %%
from pathlib import Path

import njab
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


def _load_pickle(pfath, run: int):
    df = pd.read_pickle(pfath)
    df['run'] = f'run{run:02d}'
    df = df.set_index('run', append=True)
    return df


df_long_qvalues = pd.concat(
    [_load_pickle(f, i) for i, f in enumerate(pickled_qvalues)]
)
df_long_qvalues

# %% [markdown]
# Q-values for features across runs

# %%
qvalue_stats = df_long_qvalues.groupby(level=0).agg(['mean', 'std'])
qvalue_stats.to_excel(writer,
                      sheet_name='all_qvalue_stats',
                      float_format='%3.5f')
qvalue_stats

# %%
decisions_da_target = snakemake.input.equality_rejected_target
decisions_da_target

# %%
da_counts = sum(pd.read_pickle(f) for f in decisions_da_target)
da_counts.to_excel(writer,
                   sheet_name='all_rejected_counts')
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
qvalue_stats

# %% [markdown]
# save more verbose index on scores, transfer to counts

# %%
da_counts = da_counts.loc[qvalue_stats.index]
# da_counts.to_excel(writer,
#                    sheet_name='different_rejected_counts')
qvalue_stats.index = da_counts.index
# qvalue_stats.to_excel(writer,
#                       sheet_name='different_qvalue_stats',
#                       float_format='%3.5f'
#                       )


# %%
da_counts = da_counts.droplevel(-1, axis=1)
da_counts

# %% [markdown]
# - case: feature omitted in original study
# - case: feature added: drop RSN as it does not make sense.
#         (or assing None value -> that's what counts)

# %%
mask_pgs_included_in_ald_study = qvalue_stats[('RSN', 'qvalue', 'mean')].notna()
mask_pgs_included_in_ald_study

# %%
# pgs included in original ald study
tab_diff_rejec_counts_old = (da_counts
                             .loc[mask_pgs_included_in_ald_study]
                             .reset_index()
                             .groupby(
                                 by=da_counts.columns.to_list())
                             .size()
                             .to_frame('N')
                             )
tab_diff_rejec_counts_old.to_excel(writer,
                                   sheet_name='tab_diff_rejec_counts_old')
tab_diff_rejec_counts_old

# %%
da_counts.loc[mask_pgs_included_in_ald_study
              ].to_excel(writer,
                         sheet_name='diff_rejec_counts_old')
qvalue_stats.loc[mask_pgs_included_in_ald_study
                 ].to_excel(writer,
                            sheet_name='diff_qvalue_stats_old',
                            float_format='%3.5f'
                            )

# %%
# new pgs
tab_diff_rejec_counts_new = (da_counts
                             .loc[~mask_pgs_included_in_ald_study]
                             .reset_index()
                             .drop('RSN', axis=1)
                             .groupby(
                                 by=[m for m in da_counts.columns if m != 'RSN'])
                             .size()
                             .to_frame('N')
                             )
tab_diff_rejec_counts_new.to_excel(writer,
                                   sheet_name='tab_diff_rejec_counts_new')
tab_diff_rejec_counts_new

# %%
da_counts.loc[~mask_pgs_included_in_ald_study
              ].to_excel(writer,
                         sheet_name='diff_rejec_counts_new')
qvalue_stats.loc[~mask_pgs_included_in_ald_study
                 ].to_excel(writer,
                            sheet_name='diff_qvalue_stats_new',
                            float_format='%3.5f'
                            )

# %%
mask_new_da_with_imp = mask_new_da_with_imputation = ((~mask_pgs_included_in_ald_study)
                                                      & (da_counts['None'] != 10))

tab_new_da_with_imp = njab.pandas.combine_value_counts(
    da_counts
    .loc[mask_new_da_with_imputation]
).fillna(0).astype(int)
tab_new_da_with_imp.index.name = 'number of reps'
tab_new_da_with_imp.columns.name = 'DA decisions by method'
tab_new_da_with_imp.to_excel(writer, sheet_name='tab_new_da_with_imp')
tab_new_da_with_imp

# %%
writer.close()
fname

# %%
