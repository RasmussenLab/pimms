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
# # K- Nearest Neighbors (KNN)

# %%
import logging

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import vaep
import vaep.model
import vaep.models as models
import vaep.nb
from vaep import sampling
from vaep.io import datasplits
from vaep.models import ae

logger = vaep.logging.setup_logger(logging.getLogger('vaep'))
logger.info("Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures


# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
folder_experiment: str = 'runs/example'  # Datasplit folder with data for experiment
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
# Machine parsed metadata from rawfile workflow
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
# training
epochs_max: int = 50  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
batch_size: int = 64  # Batch size for training (and evaluation)
cuda: bool = True  # Whether to use a GPU for training
# model
neighbors: int = 3  # number of neigherst neighbors to use
force_train: bool = True  # Force training when saved model could be used. Per default re-train model
sample_idx_position: int = 0  # position of index which is sample ID
model: str = 'KNN'  # model name
model_key: str = 'KNN_UNIQUE'  # potentially alternative key for model (grid search)
save_pred_real_na: bool = True  # Save all predictions for missing values
# metadata -> defaults for metadata extracted from machine data
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data


# Parameters
neighbors = 3
folder_experiment = "runs/rev3"
folder_data = "runs/appl_ald_data_2023_11/plasma/proteinGroups/data"
fn_rawfile_metadata = "data/ALD_study/processed/ald_metadata_cli.csv"
meta_cat_col = 'kleiner'

# %% [markdown]
# Some argument transformations

# %%
args = vaep.nb.get_params(args, globals=globals())
args = vaep.nb.args_from_dict(args)
args


# %% [markdown]
# Some naming conventions

# %%
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'


# %% [markdown]
# load meta data for splits


# %% [markdown]
# ## Load data in long format

# %%
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format)

# %% [markdown]
# data is loaded in long format

# %%
data.train_X.sample(5)

# %%
if args.fn_rawfile_metadata:
    df_meta = pd.read_csv(args.fn_rawfile_metadata, index_col=0)
    df_meta = df_meta.loc[data.train_X.index.levels[0]]
else:
    df_meta = None
df_meta


# %%
df_meta['to_stratify'] = df_meta[args.meta_cat_col].fillna(-1)
data.to_wide_format()
train_idx, val_test_idx = train_test_split(data.train_X.index,
                                           test_size=.2,
                                           stratify=df_meta['to_stratify'],
                                           random_state=42)
val_idx, test_idx = train_test_split(val_test_idx,
                                     test_size=.5,
                                     stratify=df_meta.loc[val_test_idx, 'to_stratify'],
                                     random_state=42)
print("Train:", train_idx.shape, "Val:", val_idx.shape, "Test:", test_idx.shape)

# %%
data.train_X.update(data.val_y.loc[train_idx])
data.train_X.update(data.test_y.loc[train_idx])
data.val_X = data.train_X.loc[val_idx]
data.test_X = data.train_X.loc[test_idx]
data.train_X = data.train_X.loc[train_idx]

data.val_y = data.val_y.loc[val_idx]
data.test_y = data.test_y.loc[test_idx]

# %%
data.to_long_format()

# %% [markdown]
# ## Initialize Comparison

# %%
freq_feat = sampling.frequency_by_index(data.train_X, 0)
freq_feat.head()  # training data

# %% [markdown]
# ### Simulated missing values

# %% [markdown]
# The validation fake NA is used to by all models to evaluate training performance.

# %%
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na

# %%
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()


# %% [markdown]
# ## Data in wide format

# %%
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X

# %% [markdown]
# ## Train
# model = 'sklearn_knn'

# %%
knn_imputer = sklearn.impute.KNNImputer(n_neighbors=args.neighbors).fit(data.train_X)

# %% [markdown]
# ### Predictions
#
# - data of training data set and validation dataset to create predictions is the same as training data.
# - predictions include missing values (which are not further compared)
#
# create predictions and select for split entries

# %%
pred = knn_imputer.transform(data.val_X)
pred = pd.DataFrame(pred, index=data.val_X.index, columns=data.val_X.columns).stack()
pred

# %%
val_pred_fake_na[args.model_key] = pred
val_pred_fake_na

# %%
pred = knn_imputer.transform(data.test_X)
pred = pd.DataFrame(pred, index=data.test_X.index, columns=data.test_X.columns).stack()

test_pred_fake_na[args.model_key] = pred
test_pred_fake_na

# %% [markdown]
# save missing values predictions

# %%
df_complete = pd.concat([data.train_X, data.val_X, data.test_X])
pred = knn_imputer.transform(df_complete)
pred = pd.DataFrame(pred, index=df_complete.index, columns=df_complete.columns).stack()
pred

# %%
if args.save_pred_real_na:
    pred_real_na = ae.get_missing_values(df_train_wide=df_complete,
                                         val_idx=val_pred_fake_na.index,
                                         test_idx=test_pred_fake_na.index,
                                         pred=pred)
    display(pred_real_na)
    pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")


# %% [markdown]
# ### Plots
#
# - validation data

# %%

# %% [markdown]
# ## Comparisons
#
# > Note: The interpolated values have less predictions for comparisons than the ones based on models (CF, DAE, VAE)
# > The comparison is therefore not 100% fair as the interpolated samples will have more common ones (especailly the sparser the data)
# > Could be changed.

# %% [markdown]
# ### Validation data
#
# - all measured (identified, observed) peptides in validation data
#
# > Does not make to much sense to compare collab and AEs,
# > as the setup differs of training and validation data differs

# %%
# papermill_description=metrics
d_metrics = models.Metrics()

# %% [markdown]
# The fake NA for the validation step are real test data (not used for training nor early stopping)

# %%
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics

# %% [markdown]
# ### Test Datasplit
#
# Fake NAs : Artificially created NAs. Some data was sampled and set
# explicitly to misssing before it was fed to the model for
# reconstruction.

# %%
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics

# %% [markdown]
# Save all metrics as json

# %%
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / f'metrics_{args.model_key}.json')
d_metrics

# %%
metrics_df = models.get_df_from_nested_dict(d_metrics.metrics,
                                            column_levels=['model', 'metric_name']).T
metrics_df

# %% [markdown]
# ## Save predictions

# %%
# save simulated missing values for both splits
val_pred_fake_na.to_csv(args.out_preds / f"pred_val_{args.model_key}.csv")
test_pred_fake_na.to_csv(args.out_preds / f"pred_test_{args.model_key}.csv")

# %% [markdown]
# ## Config

# %%
figures  # switch to fnames?

# %%
args.n_params = 1  # the number of neighbors to consider
args.dump(fname=args.out_models / f"model_config_{args.model_key}.yaml")
args

# %%
