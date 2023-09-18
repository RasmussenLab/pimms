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
# # Join aggregated metrics from each model

# %%
import pandas as pd

N_HEADER_COLS = 3
POS_INDEX_COL = 0

# %%
filepaths_in = snakemake.input
filepath_out = snakemake.output[0]
filepaths_in

# %%
filepath_out

# %% [markdown]
# ## Example
#
# - first file

# %%


def process(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath, index_col=POS_INDEX_COL, header=list(range(N_HEADER_COLS)))
    return df


process(filepaths_in[0])

# %% [markdown]
# ## Load all model metrics

# %%
metrics = pd.concat([
    process(fpath) for fpath in filepaths_in
])
metrics.stack('model')

# %%
metrics

# %% [markdown]
# ## Dump combined to disk

# %%
_ = metrics.to_csv(filepath_out)
filepath_out
