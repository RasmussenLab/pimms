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
#     name: python3
# ---

# %% [markdown]
# # Support of dumped data

# %%
import numpy as np
import pandas as pd
import plotly.express as px

import vaep  # set formatting defaults

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
# Path to json support file
support_json: str = 'data\\dev_datasets\\df_intensities_proteinGroups_long\\Q_Exactive_HF_X_Orbitrap_6070_support.json'

# %% [markdown]
# ## Completeness of samples

# %%
support = pd.read_json(support_json, typ='series').sort_values().to_frame('no. of features')
support.head()

# %%
support.describe(percentiles=np.linspace(0.1, 1, 10))

# %%
ax = support.plot(rot=90, figsize=(20, 10), legend=False)
ax.set_ylabel('number of features')
ax.yaxis.set_major_formatter("{x:,.0f}")

# %%
px.line(support, height=1000)

# %% [markdown]
# The one with very few identification are mainly fractions of entire samples
