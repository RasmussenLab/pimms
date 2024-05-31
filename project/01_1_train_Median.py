# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Variational Autoencoder

# %% tags=["hide-input"]
import logging

import pandas as pd
from IPython.display import display

import vaep
import vaep.model
import vaep.models as models
import vaep.nb
from vaep.io import datasplits

logger = vaep.logging.setup_logger(logging.getLogger('vaep'))
logger.info("Median Imputation")

figures = {}  # collection of ax or figures


# %% tags=["hide-input"]
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
folder_experiment: str = 'runs/example'  # Datasplit folder with data for experiment
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'  # Metadata for samples
# model
sample_idx_position: int = 0  # position of index which is sample ID
model_key: str = 'Median'  # model key (lower cased version will be used for file names)
model: str = 'Median'  # model name
save_pred_real_na: bool = True  # Save all predictions for real na
# metadata -> defaults for metadata extracted from machine data
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data


# %% [markdown]
# Some argument transformations


# %% tags=["hide-input"]
args = vaep.nb.get_params(args, globals=globals())
args

# %% tags=["hide-input"]
args = vaep.nb.args_from_dict(args)
args


# %% [markdown]
# Some naming conventions

# %% tags=["hide-input"]
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'

# %% [markdown]
# ## Load data in long format

# %% tags=["hide-input"]
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format)

# %% [markdown]
# data is loaded in long format

# %% tags=["hide-input"]
data.train_X.sample(5)

# %% [markdown]
# Infer index names from long format

# %% tags=["hide-input"]
index_columns = list(data.train_X.index.names)
sample_id = index_columns.pop(args.sample_idx_position)
if len(index_columns) == 1:
    index_column = index_columns.pop()
    index_columns = None
    logger.info(f"{sample_id = }, single feature: {index_column = }")
else:
    logger.info(f"{sample_id = }, multiple features: {index_columns = }")

if not index_columns:
    index_columns = [sample_id, index_column]
else:
    raise NotImplementedError("More than one feature: Needs to be implemented. see above logging output.")

# %% [markdown]
# load meta data for splits

# %% tags=["hide-input"]
if args.fn_rawfile_metadata:
    df_meta = pd.read_csv(args.fn_rawfile_metadata, index_col=0)
    display(df_meta.loc[data.train_X.index.levels[0]])
else:
    df_meta = None

# %% [markdown]
# ## Initialize Comparison
#
# - replicates idea for truely missing values: Define truth as by using n=3 replicates to impute
#   each sample
# - real test data:
#     - Not used for predictions or early stopping.
#     - [x] add some additional NAs based on distribution of data

# %% tags=["hide-input"]
freq_feat = vaep.io.datasplits.load_freq(args.data)
freq_feat.head()  # training data

# %% [markdown]
# ### Produce some addional fake samples

# %% [markdown]
# The validation fake NA is used to by all models to evaluate training performance.

# %% tags=["hide-input"]
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na

# %% tags=["hide-input"]
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()


# %% [markdown]
# ## Data in wide format
#
# - Autoencoder need data in wide format

# %% tags=["hide-input"]
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()


# %% [markdown]
# ### Add interpolation performance

# %% tags=["hide-input"]
# interpolated = vaep.pandas.interpolate(wide_df = data.train_X)
# val_pred_fake_na['interpolated'] = interpolated
# test_pred_fake_na['interpolated'] = interpolated
# del interpolated
# test_pred_fake_na

# %% tags=["hide-input"]
# Add median pred performance
args.n_params = data.train_X.shape[-1]
medians_train = data.train_X.median()
medians_train.name = args.model
pred = medians_train

val_pred_fake_na = val_pred_fake_na.join(medians_train)
test_pred_fake_na = test_pred_fake_na.join(medians_train)
val_pred_fake_na


# %% tags=["hide-input"]
if args.save_pred_real_na:
    mask = data.train_X.isna().stack()
    idx_real_na = mask.index[mask]
    idx_real_na = (idx_real_na
                   .drop(val_pred_fake_na.index)
                   .drop(test_pred_fake_na.index))
    # hacky, but works:
    pred_real_na = (pd.Series(0, index=idx_real_na, name='placeholder')
                    .to_frame()
                    .join(medians_train)
                    .drop('placeholder', axis=1))
    # pred_real_na.name = 'intensity'
    display(pred_real_na)
    pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")


# %% [markdown]
# ### Plots
#
# %% tags=["hide-input"]
feat_freq_val = val_pred_fake_na['observed'].groupby(level=-1).count()
feat_freq_val.name = 'freq_val'
ax = feat_freq_val.plot.box()

# %% tags=["hide-input"]
# # scatter plot between overall feature freq and split freq
# freq_feat.to_frame('overall').join(feat_freq_val).plot.scatter(x='overall', y='freq_val')

# %% tags=["hide-input"]
feat_freq_val.value_counts().sort_index().head()  # require more than one feat?

# %% tags=["hide-input"]
errors_val = val_pred_fake_na.drop('observed', axis=1).sub(val_pred_fake_na['observed'], axis=0)
errors_val = errors_val.abs().groupby(level=-1).mean()
errors_val = errors_val.join(freq_feat).sort_values(by='freq', ascending=True)


errors_val_smoothed = errors_val.copy()  # .loc[feat_freq_val > 1]
errors_val_smoothed[errors_val.columns[:-
                                       1]] = errors_val[errors_val.columns[:-
                                                                           1]].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x='freq', figsize=(15, 10))
# errors_val_smoothed

# %% tags=["hide-input"]
errors_val = val_pred_fake_na.drop('observed', axis=1).sub(val_pred_fake_na['observed'], axis=0)
errors_val.abs().groupby(level=-1).agg(['mean', 'count'])

# %% tags=["hide-input"]
errors_val

# %% [markdown]
# ## Comparisons

# %% [markdown]
# ### Validation data
#

# %% tags=["hide-input"]
# papermill_description=metrics
d_metrics = models.Metrics()

# %% [markdown]
# The fake NA for the validation step are real test data (not used for training nor early stopping)

# %% tags=["hide-input"]
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics

# %% [markdown]
# ### Test Datasplit
#
# Fake NAs : Artificially created NAs. Some data was sampled and set
# explicitly to misssing before it was fed to the model for
# reconstruction.

# %% tags=["hide-input"]
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics

# %% [markdown]
# The fake NA for the validation step are real test data (not used for training nor early stopping)

# %% tags=["hide-input"]

# %% [markdown]
# ### Save all metrics as json

# %% tags=["hide-input"]
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / f'metrics_{args.model_key}.json')
d_metrics


# %% tags=["hide-input"]
metrics_df = models.get_df_from_nested_dict(d_metrics.metrics, column_levels=['model', 'metric_name']).T
metrics_df

# %% [markdown]
# ## Save predictions

# %% tags=["hide-input"]
# val
fname = args.out_preds / f"pred_val_{args.model_key}.csv"
setattr(args, fname.stem, fname.as_posix())  # add [] assignment?
val_pred_fake_na.to_csv(fname)
# test
fname = args.out_preds / f"pred_test_{args.model_key}.csv"
setattr(args, fname.stem, fname.as_posix())
test_pred_fake_na.to_csv(fname)

# %% [markdown]
# ## Config

# %% tags=["hide-input"]
figures  # switch to fnames?

# %% tags=["hide-input"]
args.dump(fname=args.out_models / f"model_config_{args.model_key}.yaml")
args
