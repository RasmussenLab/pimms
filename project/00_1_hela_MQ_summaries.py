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
#     name: vaep
# ---

# %% [markdown]
# # Analysis of `summaries.txt` information
#
# - number of raw files (no here)
# - number of raw files with MQ-Output
# - MS1 per file
# - MS2 per file

# %%
import ipywidgets as widgets
import yaml
import numpy as np
from vaep.io.data_objects import MqAllSummaries
import vaep

from config import FN_ALL_SUMMARIES
print(f"{FN_ALL_SUMMARIES = }")

# %% tags=["parameters"]
FN_ALL_SUMMARIES: str = 'data/processed/all_summaries.json' # MqAllSummaries json

# %%
FN_ALL_SUMMARIES = FN_ALL_SUMMARIES

# %%
mq_all_summaries = MqAllSummaries(FN_ALL_SUMMARIES)
mq_all_summaries.df.describe().T

# %% [markdown]
# Find unique columns, see [post](https://stackoverflow.com/a/54405767/9684872)

# %%
from vaep.pandas import unique_cols
unique_cols(mq_all_summaries.df.Multiplicity), unique_cols(
    mq_all_summaries.df["Variable modifications first search"])  # int, NA

# %%
from vaep.pandas import get_unique_non_unique_columns
columns = get_unique_non_unique_columns(mq_all_summaries.df)
mq_all_summaries.df[columns.unique]

# %%
mq_all_summaries.df[columns.unique].dtypes

# %%
mq_all_summaries.df[columns.unique].iloc[0, :]


# %% [markdown]
# ## Analysis of completeness

# %%
class col_summary:
    MS1 = 'MS'
    MS2 = 'MS/MS' 
    MS2_identified  = 'MS/MS Identified'
    peptides_identified = 'Peptide Sequences Identified'

if mq_all_summaries.df is None:
    raise ValueError("No data assigned")
    
MS_spectra = mq_all_summaries.df[[col_summary.MS1, col_summary.MS2, col_summary.MS2_identified, col_summary.peptides_identified]]

def compute_summary(threshold_identified):
    mask  = MS_spectra[col_summary.peptides_identified] >= threshold_identified
    display(MS_spectra.loc[mask].describe(np.linspace(0.05, 0.95, 10)))

w_ions_range = widgets.IntSlider(value=15_000, min=.0, max=MS_spectra[col_summary.peptides_identified].max())
display(widgets.interactive(compute_summary, threshold_identified=w_ions_range))

# %% [markdown]
# List of samples without any identified peptides:

# %%
mask = (MS_spectra < 1).any(axis=1)
MS_spectra.loc[mask]

# %% [markdown]
# ## Export selected list of quantified samples
#
# Based on threshold, save a list of the specified samples

# %%
dump_dict = {'threshold': int(w_ions_range.value)}
mask = MS_spectra[col_summary.peptides_identified] >= w_ions_range.value
dump_dict['files'] = MS_spectra.loc[mask].index.to_list()

with open('data/samples_selected.yaml', 'w') as f:
    yaml.dump(dump_dict, stream=f)

# %%
