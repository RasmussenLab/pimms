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
# # Transfer predictions from NAGuideR
#

# %% tags=["hide-input"]
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import vaep
import vaep.models
import vaep.pandas
from vaep.io import datasplits

vaep.plotting.make_large_descriptors(5)

logger = vaep.logging.setup_logger(logging.getLogger('vaep'))

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
file_format: str = 'csv'  # file format of create splits, default pickle (csv)
identifer_str: str = '_all_'  # identifier for prediction files to be filtered
dumps: list = None  # list of dumps to be used

# %% [markdown]
# Some argument transformations


# %% tags=["hide-input"]
args = vaep.nb.get_params(args, globals=globals())
args = vaep.nb.args_from_dict(args)
args

# %% tags=["hide-input"]
files_out = {}

# %% [markdown]
# load data splits

# %% tags=["hide-input"]
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)


# %% [markdown]
# Validation and test data split of simulated missing values

# %% tags=["hide-input"]
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na

# %% tags=["hide-input"]
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()

# %% tags=["hide-input"]
# Find and load prediction files, filter for validation and test data

# %% tags=["hide-input"]
if args.dumps is not None:
    entire_pred = [Path(s) for s in args.dumps.split(',')]
else:
    entire_pred = list(file for file in args.out_preds.iterdir()
                       if '_all_' in str(file))
entire_pred

# %% tags=["hide-input"]
mask = data.train_X.unstack().isna().stack()
idx_real_na = mask.index[mask]
idx_real_na = (idx_real_na
               .drop(val_pred_fake_na.index)
               .drop(test_pred_fake_na.index))

for fpath in entire_pred:
    logger.info(f"Load {fpath = }")
    col_name = fpath.stem.split('_all_')[-1]
    pred = pd.read_csv(fpath, index_col=[1, 0])
    val_pred_fake_na[col_name] = pred
    fname = args.out_preds / f'pred_val_{col_name}.csv'
    files_out[fname.name] = fname.as_posix()
    val_pred_fake_na[['observed', col_name]].to_csv(fname)
    logger.info(f"Save {fname = }")

    test_pred_fake_na[col_name] = pred
    fname = args.out_preds / f'pred_test_{col_name}.csv'
    files_out[fname.name] = fname.as_posix()
    test_pred_fake_na[['observed', col_name]].to_csv(fname)
    logger.info(f"Save {fname = }")
    # hacky, but works:
    pred_real_na = (pd.Series(0, index=idx_real_na, name='placeholder')
                    .to_frame()
                    .join(pred, how='left')
                    .drop('placeholder', axis=1))
    # pred_real_na.name = 'intensity'
    fname = args.out_preds / f'pred_real_na_{col_name}.csv'
    files_out[fname.name] = fname.as_posix()
    pred_real_na.to_csv(fname)
    logger.info(f"Save {fname = }")

# del pred
# %% tags=["hide-input"]
val_pred_fake_na

# %% [markdown]
# Metrics for simulated missing values (NA)

# %% tags=["hide-input"]
# papermill_description=metrics
d_metrics = vaep.models.Metrics()

# %% tags=["hide-input"]
added_metrics = d_metrics.add_metrics(val_pred_fake_na.dropna(how='all', axis=1), 'valid_fake_na')
pd.DataFrame(added_metrics)

# %% [markdown]
# ## Test Datasplit

# %% tags=["hide-input"]
added_metrics = d_metrics.add_metrics(test_pred_fake_na.dropna(how='all', axis=1), 'test_fake_na')
pd.DataFrame(added_metrics)

# %% tags=["hide-input"]
metrics_df = vaep.models.get_df_from_nested_dict(
    d_metrics.metrics, column_levels=['model', 'metric_name']).T
metrics_df

# %% tags=["hide-input"]
order_methods = metrics_df.loc[pd.IndexSlice[:,
                                             'MAE'], 'valid_fake_na'].sort_values()
order_methods

# %% tags=["hide-input"]
top_5 = ['observed', *order_methods.droplevel(-1).index[:6]]
top_5

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(8, 2))
ax, errors_bind = vaep.plotting.errors.plot_errors_binned(
    val_pred_fake_na[top_5],
    ax=ax,
)
fname = args.out_figures / 'NAGuideR_errors_per_bin_val.png'
files_out[fname.name] = fname.as_posix()
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
files_out
