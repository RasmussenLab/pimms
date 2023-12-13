# %%
from pathlib import Path
import numpy as np
import pandas as pd


# %%
fname = 'runs/appl_ald_data_2023_11/plasma/proteinGroups/01_2_performance_summary.xlsx'
ald_pg_perf = pd.read_excel(fname, sheet_name=-1, index_col=0)
ald_pg_perf.columns = pd.MultiIndex.from_tuples([('ALD protein groups', x) for x in ald_pg_perf.columns])
ald_pg_perf

# %%
file = 'runs/mnar_mcar/all_results.xlsx'
table = [pd.read_excel(file, index_col=0, header=[0, 1])]
table.append(ald_pg_perf)
table = pd.concat(table, axis=1)
table

# %%
order = (table
         .loc[:, pd.IndexSlice[ :, 'val']]
         .mean(axis=1)
         .sort_values()
         )
order

# %%
table = table.loc[order.index]
table
# %%
fname = 'runs/supp_table_performance.xlsx'
table.to_excel(fname, float_format='%.4f')
fname

# %% [markdown]
# # Compare performance for small HeLa data set

# %%
files = (f"runs/dev_dataset_small/{level}_N50_all_small.xlsx"
         for level
         in ['proteinGroups', 'peptides', 'evidence']
         )
files
# %%
table = list()
for file in files:
    df = pd.read_excel(file, index_col=0, header=[0, 1])
    table.append(df)
table = pd.concat(table, axis=1)
table

# %%
# %%
order = (table
         .loc[:, pd.IndexSlice[:, 'val']]
         .mean(axis=1)
         .sort_values()
         )
order

# %%
table = table.loc[order.index]
table = table.style.highlight_min(axis=0)
table

# %%
fname = 'runs/dev_dataset_small/supp_table_small_HeLa.xlsx'
table.to_excel(fname, float_format='%.4f')
fname

# %%
