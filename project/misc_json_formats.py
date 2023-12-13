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
# # Json Formats
#
# - object is loaded with the correct conversions (but this is re-computed)
# - can shared information be saved as "meta" information?
#
# - [`pd.json_normalize`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html) should be able to efficiently combine information

# %%
import pandas as pd
from vaep.io.data_objects import MqAllSummaries
from vaep.pandas import get_unique_non_unique_columns

mq_all_summaries = MqAllSummaries()

# %% [markdown]
# ## summaries.json

# %% [markdown]
# ### Table format with schema

# %%
# json format with categories
columns = get_unique_non_unique_columns(mq_all_summaries.df)
columns.unique[:2]

# %%
mq_all_summaries.df[columns.unique[:3]].dtypes

# %%
type(mq_all_summaries.df.iloc[0,3])

# %%
meta = mq_all_summaries.df[columns.unique].iloc[0].to_json(indent=4, orient='table')
# print(meta)

# %%
pd.read_json(meta, orient='table').T.convert_dtypes()

# %%
pd.read_json(meta, orient='table') # produce errors when having int columns has NaN

# %%
pd.options.display.max_columns = len(columns.non_unique)
# mq_all_summaries.df[columns.non_unique]

# %%
data = mq_all_summaries.df[columns.non_unique].iloc[0:3].to_json()
data = pd.read_json(data)
data

# %%
mq_all_summaries.fp_summaries.parent /  mq_all_summaries.fp_summaries.stem / '_meta.json'

# %%
meta = mq_all_summaries.df[columns.unique].iloc[0].to_json(indent=4)
meta = pd.read_json(meta, typ='series')
meta

# %%
for col, value in meta.items():
    data[col] = value    

# %%
data

# %% [markdown]
# ## Table schema bug
#
# - filed bug report on pandas [#40255](https://github.com/pandas-dev/pandas/issues/40255)

# %%
pd.show_versions()

# %%
pd.__version__

# %%
import traceback
import pandas 
data = {'A' : [1, 2, 2, pd.NA, 4, 8, 8, 8, 8, 9],
 'B': [pd.NA] * 10}
data = pd.DataFrame(data)
data = data.astype(pd.Int64Dtype()) # in my example I get this from data.convert_dtypes()
data_json = data.to_json(orient='table', indent=4)
try:
    pd.read_json(data_json, orient='table') #ValueError: Cannot convert non-finite values (NA or inf) to integer
except ValueError as e:
    print(e)
    traceback.print_exc()

# %%
print(data.to_string())

# %%
N = 3
meta = mq_all_summaries.df[columns.unique[:N]].iloc[0:2].reset_index(drop=True)
meta.to_dict()
