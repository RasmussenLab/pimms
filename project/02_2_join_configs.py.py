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
# # Join aggregated configs from each model

# %%
import pandas as pd

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
pd.read_csv(filepaths_in[0], index_col=0)

# %% [markdown]
# ## Load all model configs

# %%
configs = pd.concat([
    pd.read_csv(fname, index_col=0) for fname in filepaths_in
])
configs

# %% [markdown]
# ## Dump combined to disk

# %%
_ = configs.to_csv(filepath_out)
filepath_out
