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
# # Datasets
#
# Datasets are `Iterable` (through their `__getitem__` and `__len__` attribute).
# Datasets are provided to `DataLoaders` which perform the aggreation to batches.

# %%
import random
import numpy as np
import pandas as pd
import vaep.io.datasets as datasets
import vaep.utils as test_data

# %%
N, M = 15, 7
data = test_data.create_random_missing_data(N, M, prop_missing=.4)

# %% [markdown]
# ## Datasets
#
# - `PeptideDatasetInMemory`
# - `PeptideDatasetInMemoryMasked`
# - `PeptideDatasetInMemoryNoMissings`

# %% [markdown]
# ## `DatasetWithMaskAndNoTarget`

# %%
dataset = datasets.DatasetWithMaskAndNoTarget(df=pd.DataFrame(data))
for _mask, _array in dataset:
    break
_array, _mask

# %% [markdown]
# ###  `PeptideDatasetInMemory`
#
# - with duplicated target in memory

# %%
dataset = datasets.PeptideDatasetInMemory(data)
for _array, _mask, _target in dataset:
    break
_array, _mask, _target

# %%
id(_array), id(_mask), id(_target) 

# %%
_array is _target # should be true

# %%
data = test_data.create_random_missing_data(N, M, prop_missing=0.3)
dataset = datasets.PeptideDatasetInMemoryMasked(df=pd.DataFrame(data), fill_na=25.0)

for _array, _mask in dataset:
    if any(_mask):
        print(_array, _mask)
        break

# %% [markdown]
# ### `DatasetWithTarget`

# %%
data = test_data.create_random_missing_data(N, M, prop_missing=0.3)
dataset = datasets.DatasetWithTarget(df=pd.DataFrame(data))

for _mask, _array, target in dataset:
    if any(_mask):
        print(_array, _mask, target, sep='\n')
        break

# %% [markdown]
# ### `DatasetWithTargetSpecifyTarget`

# %%
data = test_data.create_random_missing_data(N, M, prop_missing=0.2)

df = pd.DataFrame(data)

val_y = df.stack().groupby(level=0).sample(frac=0.2)
# targets = val_y.unstack().sort_index()
targets = val_y.unstack()

df[targets.notna()] = pd.NA
df

# %% [markdown]
# The targets are complementary

# %%
targets

# %%
dataset = datasets.DatasetWithTargetSpecifyTarget(df=df, targets=targets)
for _mask, _array, target in dataset:
    if any(_mask):
        print(_mask, _array, target, sep='\n')
        break

# %%
row = random.randint(0,len(dataset)-1)
print(f"{row = }")
dataset[row]

# %% [markdown]
# ### `PeptideDatasetInMemoryNoMissings`

# %%
# data and pd.DataFrame.data share the same memory
try:
    dataset = datasets.PeptideDatasetInMemoryNoMissings(data)
    for _array in dataset:
        print(_array)
        break
except AssertionError as e:
    print(e)

# %% [markdown]
# ## DataLoaders
#
# FastAI DataLoaders accept pytorch datasets

# %%
from fastai.collab import CollabDataLoaders
# , MSELossFlat, Learner
# from fastai.collab import EmbeddingDotBias

from vaep.io.datasplits import long_format


data = pd.DataFrame(data)
data.index.name, data.columns.name = ('Sample ID', 'peptide')
df_long = long_format(pd.DataFrame(data))
df_long.reset_index(inplace=True)
df_long.head()

# %%
dls = CollabDataLoaders.from_df(df_long,  valid_pct=0.15, 
                                user_name='Sample ID', item_name='peptide', rating_name='intensity',
                               bs=4)
type(dls.dataset), dls.dataset._dl_type # no __mro__?

# %% [markdown]
# Iterating over the dataset gives the column names

# %%
for x in dls.dataset:
    print(x)

# %% [markdown]
# Training DataFrame is hidden under items

# %%
dls.dataset.items

# %%
for x in dls.train_ds:
    print(x)
    break

# %%
dls.train_ds

# %% [markdown]
# Iterating over the dataset returns columns, not single rows

# %%
# dls.train_ds.__getitem__??

# %%
dls.train_ds.items['Sample ID']

# %% [markdown]
# But the `DataLoader` return the numeric representation in batches:

# %%
for batch in dls.train_ds:
    break
batch

# %%
# dls.train.__iter__??

# %%
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
# _SingleProcessDataLoaderIter??

# %% [markdown]
# So.. It seems too complicated
# - the `_collate_fn` seems to aggrete the data from the DataFrame
# - should be possible to keep track of that 

# %%
next(iter(dls.dataset))


# %%
