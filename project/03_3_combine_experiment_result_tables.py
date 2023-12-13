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
# # Combine tables

# %%
from pathlib import Path
import pandas as pd

# %% [markdown]
# Use parent folder name as key

# %%
files = {Path(f).parent.name: f for f in snakemake.input}
files

# %%
table = []
for key, file in files.items():
    df = pd.read_excel(file, sheet_name=-1, index_col=0)
    df.columns = pd.MultiIndex.from_tuples([(key, x) for x in df.columns])
    table.append(df)

table = pd.concat(table, axis=1)
table

# %% [markdown]
# Order by average validation split performance

# %%
order = (table
         .loc[:, pd.IndexSlice[:, 'val']]
         .mean(axis=1)
         .sort_values()
         )
order

# %%
table = table.loc[order.index]
table

# %% [markdown]
# Save table

# %%
fname = snakemake.output.combined_xlsx
table.to_excel(fname, float_format='%.4f')
fname
