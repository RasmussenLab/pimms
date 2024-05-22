# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,tags,-all
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
# # Transfer data for NAGuideR format
#

# %% tags=["hide-input"]
import pandas as pd

import vaep
import vaep.models
from vaep.io import datasplits

# %% tags=["hide-input"]
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
# Datasplit folder with data for experiment
folder_experiment: str = 'runs/example'
folder_data: str = ''  # specify data directory if needed
file_format_in: str = 'csv'  # file format of original splits, default pickle (pkl)
file_format_out: str = 'csv'  # file format of transformed splits, default csv

# %% tags=["hide-input"]
args = vaep.nb.get_params(args, globals=globals())
args

# %% tags=["hide-input"]
params = vaep.nb.args_from_dict(args)
# params = OmegaConf.create(args)
params

# %% tags=["hide-input"]
splits = datasplits.DataSplits.from_folder(params.data, file_format=params.file_format_in)

# %% tags=["hide-input"]
train_data = splits.train_X.unstack()
train_data


# %% [markdown]
# Save placeholder sample annotation for use in NAGuideR app which requires such a file


# %% tags=["hide-input"]
annotation = pd.Series('test', train_data.index).to_frame('group')
annotation.index.name = 'Samples'
annotation

# %% tags=["hide-input"]
fname = params.data / 'sample_annotation_placeholder.csv'
annotation.to_csv(fname)
fname

# %% [markdo]
# Save with samples in columns

# %% tags=["hide-input"]
fname = params.data / 'data_wide_sample_cols.csv'
# fillna('Filtered')
train_data.T.to_csv(fname)
fname


# %% tags=["hide-input"]
# 'data_wide_sample_cols.csv'
