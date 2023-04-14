# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transfer predictions from NAGuideR

# %%
import logging
import pandas as pd
import seaborn as sns

import vaep
import vaep.models
from vaep.io import datasplits
import vaep.pandas
from vaep.pandas import calc_errors


logger = vaep.logging.setup_logger(logging.getLogger('vaep'))

# %%
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


# %%
args = vaep.nb.get_params(args, globals=globals())
args = vaep.nb.args_from_dict(args)
args

# %% [markdown]
# load data splits

# %%
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)


# %% [markdown]
# Validation and test data split of simulated missing values

# %%
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na

# %%
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()

# %%
# Find and load prediction files, filter for validation and test data

# %%
if args.dumps is not None:
    entire_pred = args.dumps.split(',')
entire_pred

# %%
entire_pred = list(file for file in args.out_preds.iterdir()
                   if '_all_' in str(file))
entire_pred

# %%
for fpath in entire_pred:
    col_name = fpath.stem.split('_all_')[-1]
    pred = pd.read_csv(fpath, index_col=[1,0])
    # pred.columns = pred.columns.str[1:].str.replace(
    #     '.', '-', regex=False)  # NaGuideR change the sample names
    # pred.columns.name = test_pred_fake_na.index.names[0]
    # pred.index.name = test_pred_fake_na.index.names[1]
    # pred = pred.unstack()

    val_pred_fake_na[col_name] = pred
    val_pred_fake_na[['observed', col_name]].to_csv(
        args.out_preds / f'pred_val_{col_name}.csv')

    test_pred_fake_na[col_name] = pred
    test_pred_fake_na[['observed', col_name]].to_csv(
        args.out_preds / f'pred_test_{col_name}.csv')

# del pred
# %%
val_pred_fake_na

# %% [markdown]
# Metrics for simulated missing values (NA)

# %%
# papermill_description=metrics
d_metrics = vaep.models.Metrics()

# %%
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics

# %% [markdown]
# ### Test Datasplit

# %%
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics

# %%
metrics_df = vaep.models.get_df_from_nested_dict(
    d_metrics.metrics, column_levels=['model', 'metric_name']).T
metrics_df

errors = calc_errors.calc_errors_per_bin(val_pred_fake_na, target_col='observed')
errors

# %%
top5 = errors.drop(['bin', 'n_obs'], axis=1).mean().sort_values().iloc[:5].index.to_list()
errors[top5].describe()

# %%
meta_cols = ['bin', 'n_obs']
n_obs = errors[meta_cols].apply(
        lambda x: f"{x.bin} (N={x.n_obs:,d})", axis=1
        ).rename('bin').astype('category')

errors_long = (errors[top5]
               #.drop(meta_cols, axis=1)
               .stack()
               .to_frame('intensity')
               .join(n_obs)
               .reset_index()
)
errors_long.sample(5)

# %%
ax = sns.barplot(data=errors_long,
            x='bin', y='intensity', hue='model')
ax.xaxis.set_tick_params(rotation=-90)

fname = args.out_figures / 'NAGuideR_errors_per_bin.png'
vaep.savefig(ax.get_figure(), fname)

# %%
