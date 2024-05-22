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
# # Imputation using random draws from shifted normal distribution

# %% tags=["hide-input"]
import logging

import pandas as pd
from IPython.display import display

import vaep
import vaep.imputation
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
# Datasplit folder with data for experiment
folder_experiment: str = 'runs/example'
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
# Machine parsed metadata from rawfile workflow
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
# model
sample_idx_position: int = 0  # position of index which is sample ID
# model key (lower cased version will be used for file names)
axis: int = 1  # impute per row/sample (1) or per column/feat (0).
completeness = 0.6  # fractio of non missing values for row/sample (axis=0) or column/feat (axis=1)
model_key: str = 'RSN'
model: str = 'RSN'  # model name
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
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)

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
    raise NotImplementedError(
        "More than one feature: Needs to be implemented. see above logging output.")

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

# %% tags=["hide-input"]
freq_feat = vaep.io.datasplits.load_freq(args.data)
freq_feat.head()  # training data

# %% [markdown]
# ### Produce some addional fake samples

# %% [markdown]
# The validation simulated NA is used to by all models to evaluate training performance.

# %% tags=["hide-input"]
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na

# %% tags=["hide-input"]
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()

# %% [markdown]
# ## Data in wide format

# %% tags=["hide-input"]
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()


# %% [markdown]
# ### Impute using shifted normal distribution

# %% tags=["hide-input"]
imputed_shifted_normal = vaep.imputation.impute_shifted_normal(
    data.train_X,
    mean_shift=1.8,
    std_shrinkage=0.3,
    completeness=args.completeness,
    axis=args.axis)
imputed_shifted_normal = imputed_shifted_normal.to_frame('intensity')
imputed_shifted_normal

# %% tags=["hide-input"]
val_pred_fake_na[args.model] = imputed_shifted_normal
test_pred_fake_na[args.model] = imputed_shifted_normal
val_pred_fake_na

# %% [markdown]
# Save predictions for NA

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
                    .join(imputed_shifted_normal)
                    .drop('placeholder', axis=1))
    # pred_real_na.name = 'intensity'
    display(pred_real_na)
    pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")


# # %% [markdown]
# ### Plots
#
# %% tags=["hide-input"]
ax, _ = vaep.plotting.errors.plot_errors_binned(val_pred_fake_na)

# %% tags=["hide-input"]
ax, _ = vaep.plotting.errors.plot_errors_binned(test_pred_fake_na)

# %% [markdown]
# ## Comparisons


# %% [markdown]
# ### Validation data
#
# - all measured (identified, observed) peptides in validation data

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
# The fake NA for the validation step are real test data

# %% [markdown]
# ### Save all metrics as json

# %% tags=["hide-input"]
vaep.io.dump_json(d_metrics.metrics, args.out_metrics /
                  f'metrics_{args.model_key}.json')
d_metrics

# %% tags=["hide-input"]
metrics_df = models.get_df_from_nested_dict(
    d_metrics.metrics, column_levels=['model', 'metric_name']).T
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
