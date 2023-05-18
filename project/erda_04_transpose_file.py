# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transpose file

# %%
from pathlib import Path
import pandas as pd

import vaep

import config

# %% [markdown]
# Paramters

# %%
# out_folder = Path('data/selected/proteinGroups') 
# fname = out_folder / 'intensities_wide_selected_N04550_M07444.pkl'

# out_folder = Path('data/selected/peptides') 
# fname = out_folder / 'intensities_wide_selected_N42881_M07441.pkl'

out_folder = Path('data/selected/evidence') 
fname = out_folder / 'intensities_wide_selected_N49560_M07444.pkl'


# %%
def get_template(fname, split='_N'):
    ext = fname.suffix
    stem = fname.stem.split(split)[0]
    return f"{stem}{{}}{ext}"

def memory_usage_in_mb(df):
    return df.memory_usage(deep=True).sum() / (2**20)

template = get_template(fname)
template

# %%
files_out = {}

# %%
# %%time
df = pd.read_pickle(fname)
df.head()

# %%
df.memory_usage(deep=True).sum() / (2**20)

# %% [markdown]
# Here reading the csv file is slightly faster and consumes less memory.
#
# - dtype: `float64` -> missing values as `np.nan`
# - but: saving to csv will be larger.

# %%
# # %%time
# df = pd.read_csv(fname.with_suffix('.csv'), index_col=0)
# df.memory_usage(deep=True).sum() / (2**20) 

# %%
# %%time
count_samples = df.notna().sum()

fname = out_folder / 'count_samples.json'
count_samples.to_json(fname)

vaep.plotting.make_large_descriptors(size='medium')

ax = count_samples.sort_values().plot(rot=90, ylabel='observations')
ax.yaxis.set_major_formatter("{x:,.0f}")
vaep.savefig(ax.get_figure(), fname)

# %%
# %%time
df = df.T
df.memory_usage(deep=True).sum() / (2**20)

# %%
# %%time
fname = out_folder / config.insert_shape(df, template=template)
files_out[fname.name] = fname.as_posix()
df.to_pickle(fname)

# %%
# %%time
fname = fname.with_suffix('.csv')
files_out[fname.name] = fname.as_posix()
df.to_csv(fname, chunksize=1_000)

# %%
count_features = df.notna().sum()
fname = out_folder / 'count_feat.json'
count_features.to_json(fname)

ax = count_features.sort_values().plot(rot=90, ylabel='observations')
ax.yaxis.set_major_formatter("{x:,.0f}")
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ## Present abesent pattern

# %%
df = df.notna().astype(pd.Int8Dtype())
df

# %%
# %%time
fname = out_folder / config.insert_shape(df,  'absent_0_present_1_selected{}.pkl')

files_out[fname.name] = fname.as_posix()
df.to_pickle(fname)

# %%
files_outfname = fname.with_suffix('.csv')
files_out[fname.name] = fname.as_posix()
df.replace(0, pd.NA).to_csv(fname.with_suffix('.csv'), chunksize=1_000)

# %% [markdown]
# ## Files written

# %%
files_out
