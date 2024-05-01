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

# %%

import logging
from functools import partial

from fastai.basics import *
from fastai.learner import Learner
from fastai.callback.all import *
from fastai.callback.all import EarlyStoppingCallback
from fastai.torch_basics import *

import torch

from IPython.display import display

from torch.nn import Sigmoid

import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import vaep
import vaep.nb
from vaep.io import datasplits
from vaep.models import ae
import vaep.models as models
import vaep.model
from vaep.analyzers import analyzers


# overwriting Recorder callback with custom plot_loss
from vaep.models import plot_loss
from fastai import learner
learner.Recorder.plot_loss = plot_loss


logger = vaep.logging.setup_logger(logging.getLogger('vaep'))
logger.info(
    "Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures


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
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
# Machine parsed metadata from rawfile workflow
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
# training
epochs_max: int = 50  # Maximum number of epochs
batch_size: int = 64  # Batch size for training (and evaluation)
cuda: bool = True  # Whether to use a GPU for training
# model
# Dimensionality of encoding dimension (latent space of model)
latent_dim: int = 25
# A underscore separated string of layers, '256_128' for the encoder, reverse will be use for decoder
hidden_layers: str = '256_128'
# force_train:bool = True # Force training when saved model could be used. Per default re-train model
patience: int = 50  # Patience for early stopping
sample_idx_position: int = 0  # position of index which is sample ID
model: str = 'VAE'  # model name
model_key: str = 'VAE'  # potentially alternative key for model (grid search)
save_pred_real_na: bool = True  # Save all predictions for missing values
# metadata -> defaults for metadata extracted from machine data
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data

# %% [markdown]
# Some argument transformations


# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
args = vaep.nb.args_from_dict(args)

if isinstance(args.hidden_layers, str):
    args.overwrite_entry("hidden_layers", [int(x)
                         for x in args.hidden_layers.split('_')])
else:
    raise ValueError(
        f"hidden_layers is of unknown type {type(args.hidden_layers)}")
args


# %% [markdown]
# Some naming conventions

# %%
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'

# %% [markdown]
# ## Load data in long format

# %%
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)

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
    raise NotImplementedError(
        "More than one feature: Needs to be implemented. see above logging output.")

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
freq_feat.head()  # training data

# %% [markdown]
# ### Produce some addional simulated samples

# %% [markdown]
# The validation simulated NA is used to by all models to evaluate training performance.

# %%
val_pred_simulated_na = data.val_y.to_frame(name='observed')
val_pred_simulated_na

# %%
test_pred_simulated_na = data.test_y.to_frame(name='observed')
test_pred_simulated_na.describe()


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

# %% [markdown]
# ### Fill Validation data with potentially missing features

# %%
data.train_X

# %%
data.val_y  # potentially has less features

# %%
data.val_y = pd.DataFrame(pd.NA, index=data.train_X.index,
                          columns=data.train_X.columns).fillna(data.val_y)
data.val_y

# %% [markdown]
# ## Variational Autoencoder

# %% [markdown]
# ### Analysis: DataLoaders, Model, transform

# %%
default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])

# %% [markdown]
# ### Analysis: DataLoaders, Model

# %%

analysis = ae.AutoEncoderAnalysis(  # datasplits=data,
    train_df=data.train_X,
    val_df=data.val_y,
    model=models.vae.VAE,
    model_kwargs=dict(n_features=data.train_X.shape[-1],
                      n_neurons=args.hidden_layers,
                      # last_encoder_activation=None,
                      last_decoder_activation=None,
                      dim_latent=args.latent_dim),
    transform=default_pipeline,
    decode=['normalize'],
    bs=args.batch_size)
args.n_params = analysis.n_params_ae
if args.cuda:
    analysis.model = analysis.model.cuda()
analysis.model

# %% [markdown]
# ### Training
#
#

# %%
results = []
loss_fct = partial(models.vae.loss_fct, results=results)

# %%
analysis.learn = Learner(dls=analysis.dls,
                         model=analysis.model,
                         loss_func=loss_fct,
                         cbs=[ae.ModelAdapterVAE(),
                              EarlyStoppingCallback(patience=args.patience)
                              ])

analysis.learn.show_training_loop()
# %% [markdown]
# Adding a `EarlyStoppingCallback` results in an error.  Potential fix in
# [PR3509](https://github.com/fastai/fastai/pull/3509) is not yet in
# current version. Try again later

# %%
# learn.summary()

# %%
suggested_lr = analysis.learn.lr_find()
analysis.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr

# %%
results.clear()  # reset results

# %% [markdown]
# dump model config

# %%
# needs class as argument, not instance, but serialization needs instance
analysis.params['last_decoder_activation'] = Sigmoid()

vaep.io.dump_json(
    vaep.io.parse_dict(
        analysis.params, types=[
            (torch.nn.modules.module.Module, lambda m: str(m))
        ]),
    args.out_models / TEMPLATE_MODEL_PARAMS.format(args.model_key))

# restore original value
analysis.params['last_decoder_activation'] = Sigmoid

# %%
# papermill_description=train
analysis.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)

# %% [markdown]
# Save number of actually trained epochs

# %%
args.epoch_trained = analysis.learn.epoch + 1
args.epoch_trained

# %% [markdown]
# #### Loss normalized by total number of measurements

# %%
N_train_notna = data.train_X.notna().sum().sum()
N_val_notna = data.val_y.notna().sum().sum()
fig = models.plot_training_losses(analysis.learn, args.model_key,
                                  folder=args.out_figures,
                                  norm_factors=[N_train_notna, N_val_notna])

# %% [markdown]
# ### Predictions
# create predictions and select validation data predictions

# %%
analysis.model.eval()
pred, target = res = ae.get_preds_from_df(df=data.train_X, learn=analysis.learn,
                                          position_pred_tuple=0,
                                          transformer=analysis.transform)
pred = pred.stack()
pred

# %%
val_pred_simulated_na['VAE'] = pred  # 'model_key' ?
val_pred_simulated_na


# %%
test_pred_simulated_na['VAE'] = pred  # model_key?
test_pred_simulated_na

# %% [markdown]
# save missing values predictions

# %%
if args.save_pred_real_na:
    pred_real_na = ae.get_missing_values(df_train_wide=data.train_X,
                                         val_idx=val_pred_simulated_na.index,
                                         test_idx=test_pred_simulated_na.index,
                                         pred=pred)
    display(pred_real_na)
    pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")


# %% [markdown]
# ### Plots
#
# - validation data

# %%
analysis.model = analysis.model.cpu()
df_latent = vaep.model.get_latent_space(analysis.model.get_mu_and_logvar,
                                        dl=analysis.dls.valid,
                                        dl_index=analysis.dls.valid.data.index)
df_latent

# %%
ana_latent = analyzers.LatentAnalysis(df_latent,
                                      df_meta,
                                      args.model_key,
                                      folder=args.out_figures)
if args.meta_date_col and df_meta is not None:
    figures[f'latent_{args.model_key}_by_date'], ax = ana_latent.plot_by_date(
        args.meta_date_col)

# %%
if args.meta_cat_col and df_meta is not None:
    figures[f'latent_{args.model_key}_by_{"_".join(args.meta_cat_col.split())}'], ax = ana_latent.plot_by_category(
        args.meta_cat_col)

# %%
feat_freq_val = val_pred_simulated_na['observed'].groupby(level=-1).count()
feat_freq_val.name = 'freq_val'
ax = feat_freq_val.plot.box()

# %%
feat_freq_val.value_counts().sort_index().head()  # require more than one feat?

# %%
errors_val = val_pred_simulated_na.drop('observed', axis=1).sub(
    val_pred_simulated_na['observed'], axis=0)
errors_val = errors_val.abs().groupby(level=-1).mean()
errors_val = errors_val.join(freq_feat).sort_values(by='freq', ascending=True)


errors_val_smoothed = errors_val.copy()  # .loc[feat_freq_val > 1]
errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]
                                                          ].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x='freq', figsize=(15, 10))
# errors_val_smoothed

# %%
errors_val = val_pred_simulated_na.drop('observed', axis=1).sub(
    val_pred_simulated_na['observed'], axis=0)
errors_val.abs().groupby(level=-1).agg(['mean', 'count'])

# %%
errors_val

# %% [markdown]
# ## Comparisons
#
# Simulated NAs : Artificially created NAs. Some data was sampled and set
# explicitly to misssing before it was fed to the model for
# reconstruction.

# %% [markdown]
# ### Validation data
#
# - all measured (identified, observed) peptides in validation data

# %%
# papermill_description=metrics
# d_metrics = models.Metrics(no_na_key='NA interpolated', with_na_key='NA not interpolated')
d_metrics = models.Metrics()

# %% [markdown]
# The simulated NA for the validation step are real test data (not used for training nor early stopping)

# %%
added_metrics = d_metrics.add_metrics(val_pred_simulated_na, 'valid_simulated_na')
added_metrics

# %% [markdown]
# ### Test Datasplit
#


# %%
added_metrics = d_metrics.add_metrics(test_pred_simulated_na, 'test_simulated_na')
added_metrics

# %% [markdown]
# Save all metrics as json

# %%
vaep.io.dump_json(d_metrics.metrics, args.out_metrics /
                  f'metrics_{args.model_key}.json')
d_metrics

# %%
metrics_df = models.get_df_from_nested_dict(
    d_metrics.metrics, column_levels=['model', 'metric_name']).T
metrics_df

# %% [markdown]
# ## Save predictions

# %%
# save simulated missing values for both splits
val_pred_simulated_na.to_csv(args.out_preds / f"pred_val_{args.model_key}.csv")
test_pred_simulated_na.to_csv(args.out_preds / f"pred_test_{args.model_key}.csv")

# %% [markdown]
# ## Config

# %%
figures  # switch to fnames?

# %%
args.dump(fname=args.out_models / f"model_config_{args.model_key}.yaml")
args
