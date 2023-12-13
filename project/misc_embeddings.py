# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Understanding Embeddings

# %%
from fastai.tabular.all import *
from fastai.collab import *

# %%
# Embedding?

# %%
Embedding

# %%
e = Embedding(100, 10)

# %%
idx = torch.tensor([1, 3])
e(idx).detach().numpy()

# %%
idx = torch.tensor([1, 2])
e(idx).detach().numpy()

# %%
