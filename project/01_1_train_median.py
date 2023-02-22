# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Variational Autoencoder

# %%
import logging
import plotly.express as px

import pandas as pd

import sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import vaep
from vaep.analyzers import analyzers
import vaep.model
import vaep.models as models
from vaep.io import datasplits

import vaep.nb
logger = vaep.logging.setup_logger(logging.getLogger('vaep'))
logger.info("Median Imputation")

figures = {}  # collection of ax or figures


# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# files and folders
folder_experiment:str = 'runs/example' # Datasplit folder with data for experiment
file_format: str = 'pkl' # file format of create splits, default pickle (pkl)
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv' # Machine parsed metadata from rawfile workflow
# model
sample_idx_position: int = 0 # position of index which is sample ID
model_key: str = 'Median' # model key (lower cased version will be used for file names)
model: str = 'Median' # model name
save_pred_real_na: bool = True # Save all predictions for real na
# metadata -> defaults for metadata extracted from machine data
meta_date_col: str = None # date column in meta data
meta_cat_col: str = None # category column in meta data


# %% [markdown]
# Some argument transformations


# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
args = vaep.nb.args_from_dict(args)
args


# %% [markdown]
# Some naming conventions

# %%
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'

# %% [markdown]
# ## Load data in long format

# %%
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format) 

# %% [markdown]
# data is loaded in long format

# %%
data.train_X.sample(5)

# %% [markdown]
# Infer index names from long format 

# %%
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

# %%
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

# %%
freq_feat = vaep.io.datasplits.load_freq(args.data)
freq_feat.head() # training data

# %% [markdown]
# ### Produce some addional fake samples

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
#
# - Autoencoder need data in wide format

# %%
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()


# %% [markdown]
# ### Add interpolation performance

# %%
interpolated = vaep.pandas.interpolate(wide_df = data.train_X) 
val_pred_fake_na['interpolated'] = interpolated
test_pred_fake_na['interpolated'] = interpolated
del interpolated
test_pred_fake_na

# %%
# Add median pred performance
medians_train = data.train_X.median()
medians_train.name = args.model
pred = medians_train

val_pred_fake_na = val_pred_fake_na.join(medians_train)
test_pred_fake_na = test_pred_fake_na.join(medians_train)
val_pred_fake_na


# # %%
# pred = pd.concat([val_pred_fake_na, test_pred_fake_na])[args.model]
# pred

# if args.save_pred_real_na:
    # # all idx missing in training data
    # mask = data.train_X.isna().stack()
    # idx_real_na = mask.index[mask]
    # # remove fake_na idx
    # idx_real_na = idx_real_na.drop(val_pred_fake_na.index).drop(test_pred_fake_na.index)
    # pred_real_na = pred.loc[idx_real_na]
    # pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")
    # del mask, idx_real_na, pred_real_na, pred

# %% [markdown]
# ### Plots
#
# %%
feat_freq_val = val_pred_fake_na['observed'].groupby(level=-1).count()
feat_freq_val.name = 'freq_val'
ax = feat_freq_val.plot.box()

# %%
# # scatter plot between overall feature freq and split freq
# freq_feat.to_frame('overall').join(feat_freq_val).plot.scatter(x='overall', y='freq_val')

# %%
feat_freq_val.value_counts().sort_index().head() # require more than one feat?

# %%
errors_val = val_pred_fake_na.drop('observed', axis=1).sub(val_pred_fake_na['observed'], axis=0)
errors_val = errors_val.abs().groupby(level=-1).mean()
errors_val = errors_val.join(freq_feat).sort_values(by='freq', ascending=True)


errors_val_smoothed = errors_val.copy() #.loc[feat_freq_val > 1]
errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x='freq', figsize=(15,10) )
# errors_val_smoothed

# %%
errors_val = val_pred_fake_na.drop('observed', axis=1).sub(val_pred_fake_na['observed'], axis=0)
errors_val.abs().groupby(level=-1).agg(['mean', 'count'])

# %%
errors_val

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
# > Does not make too much sense to compare collab and AEs,  
# > as the setup differs of training and validation data differs

# %%
# papermill_description=metrics
d_metrics = models.Metrics(no_na_key='NA interpolated', with_na_key='NA not interpolated')

# %% [markdown]
# The fake NA for the validation step are real test data (not used for training nor early stopping)

# %%
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics

# %% [markdown]
# ### Test Datasplit
#
# Fake NAs : Artificially created NAs. Some data was sampled and set explicitly to misssing before it was fed to the model for reconstruction.

# %%
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics

# %% [markdown]
# ### Save all metrics as json

# %%
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / f'metrics_{args.model_key}.json')
d_metrics


# %%
metrics_df = models.get_df_from_nested_dict(d_metrics.metrics).T
metrics_df

# %% [markdown]
# ### Plot metrics

# %%
plotly_view = metrics_df.stack().unstack(-2).set_index('N', append=True)
plotly_view.head()

# %% [markdown]
# #### Fake NA which could be interpolated
#
# - bulk of validation and test data

# %%
plotly_view.loc[pd.IndexSlice[:, :, 'NA interpolated']]

# %%
subset = 'NA interpolated'
fig = px.scatter(plotly_view.loc[pd.IndexSlice[:, :, subset]].stack().to_frame('metric_value').reset_index(),
                 x="data_split",
                 y='metric_value',
                 color="model",
                 facet_row="metric_name",
                 # facet_col="subset",
                 hover_data='N',
                 title=f'Performance for {subset}',
                 labels={"data_split": "data",
                         "metric_value": '', 'metric_name': 'metric'},
                 height=500,
                 width=300,
                 )
fig.show()

# %% [markdown]
# #### Fake NA which could not be interpolated
#
# - small fraction of total validation and test data
#
# > not interpolated fake NA values are harder to predict for models  
# > Note however: fewer predicitons might mean more variability of results

# %%
plotly_view.loc[pd.IndexSlice[:, :, 'NA not interpolated']]

# %%
subset = 'NA not interpolated'
fig = px.scatter(plotly_view.loc[pd.IndexSlice[:, :, subset]].stack().to_frame('metric_value').reset_index(),
                 x="data_split",
                 y='metric_value',
                 color="model",
                 facet_row="metric_name",
                 # facet_col="subset",
                 hover_data='N',
                 title=f'Performance for {subset}',
                 labels={"data_split": "data",
                         "metric_value": '', 'metric_name': 'metric'},
                 height=500,
                 width=300,
                 )
fig.show()

# %% [markdown]
# ## Save predictions

# %%
# val
fname = args.out_preds / f"pred_val_{args.model_key.lower()}.csv"
setattr(args, fname.stem, fname.as_posix()) # add [] assignment?
val_pred_fake_na.to_csv(fname)
# test
fname = args.out_preds / f"pred_test_{args.model_key.lower()}.csv"
setattr(args, fname.stem, fname.as_posix())
test_pred_fake_na.to_csv(fname)

# %% [markdown]
# ## Config

# %%
figures # switch to fnames?

# %%
args.dump(fname=args.out_models/ f"model_config_{args.model_key}.yaml")
args
