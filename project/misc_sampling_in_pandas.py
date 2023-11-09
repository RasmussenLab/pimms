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
# ## Sampling with weights in Pandas
#
# - sampling core utilities is based on numpy (see docstring)
# - [file](https://github.com/pandas-dev/pandas/blob/49d371364b734b47c85733aac74b03ac4400c629/pandas/core/sample.py) containing sampling functions

# %% [markdown]
# ## Some random data

# %%
from vaep.utils import create_random_df
X = create_random_df(100, 15, prop_na=0.1).stack().to_frame(
    'intensity').reset_index()

freq = X.peptide.value_counts().sort_index()
freq.name = 'freq'

X = X.set_index(keys=list(X.columns[0:2]))  # to_list as an alternative
freq

# %%
X

# %%
print(f"Based on total number of rows, 95% is roughly: {int(len(X) * 0.95)}")
print("Based on each sample's 95% obs, it is roughly: {}".format(
    X.groupby('Sample ID').apply(lambda df: int(len(df) * 0.95)).sum()))

# %% [markdown]
# ## Samling using a column with the weights

# %%
X = X.join(freq, on='peptide')
X

# %%
t = X.groupby('Sample ID').get_group('sample_003')
t

# %%
t.sample(frac=0.75, weights='freq')

# %% [markdown]
# Sampling the entire DataFrame based on the freq will normalize on N of all rows. The normalization leaves relative frequency the same (if no floating point unprecision is reached)

# %%
# number of rows not the same as when using groupby (see above)
X.sample(frac=0.95, weights='freq')

# %% [markdown]
# ### Sampling fails with groupby, reindexing needed

# %% [markdown]
# The above is not mapped one to one to the groupby sample method. One needs to apply it to every single df.

# %%
# X.groupby('Sample ID').sample(frac=0.95, weights='freq') # does not work
X.groupby('Sample ID').apply(
    lambda df: df.reset_index(0, drop=True).sample(frac=0.95, weights='freq')
).drop('freq', axis=1)

# %% [markdown]
# And passing a Series need the original X to be indexed the same (multi-indices are not supported)

# %%
# for i, t in X.groupby('Sample ID'):
#     t = t.sample(frac=0.75, weights=freq)
# t

# %%
X = X.reset_index('Sample ID')
X

# %%
X.groupby(by='Sample ID').sample(frac=0.95, weights=freq)

# %%
X.groupby(by='Sample ID').get_group('sample_002')

# %% [markdown]
# ## Sanity check: Downsampling the first feature

# %%
freq.loc['feat_00'] = 1  # none should be selected

# %%
freq = freq / freq.sum()
freq

# %%
X.groupby(by='Sample ID').sample(
    frac=0.5, weights=freq).sort_index().reset_index().peptide.value_counts()

# %% [markdown]
# ## Using a series
#
# - in the above approach, sampling weights might be readjusted based on the values present in `sample` as `NAN`s lead to the weights not summing up. Alteratively one could loop through the wide format rows and sample values from these.

# %%
freq

# %%
X = X.drop('freq', axis=1).set_index(
    'Sample ID', append=True).squeeze().unstack(0)
X

# %%
X.iloc[0].sample(frac=0.8, weights=freq).sort_index()

# %% [markdown]
# Sampling using the wide format would garuantee that the weights are not adjusted based on missing values, but that instead missing values are sample into on or the other set. Ultimately `NaN`s are dropped also in this approach.

# %%
import pandas as pd
data = {}
for row_key in X.index:
    data[row_key] = X.loc[row_key].sample(frac=0.8, weights=freq)
pd.DataFrame(data).stack()
