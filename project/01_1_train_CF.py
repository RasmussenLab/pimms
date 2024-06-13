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
# # Collaborative Filtering

# %% tags=["hide-input"]
import logging
from pprint import pprint

import matplotlib.pyplot as plt
# overwriting Recorder callback with custom plot_loss
from fastai import learner
from fastai.collab import *
from fastai.collab import (EarlyStoppingCallback, EmbeddingDotBias, Learner,
                           MSELossFlat, default_device)
from fastai.tabular.all import *

import vaep
import vaep.model
import vaep.models as models
import vaep.nb
from vaep.io import datasplits
from vaep.logging import setup_logger
from vaep.models import RecorderDump, plot_loss

learner.Recorder.plot_loss = plot_loss
# import fastai.callback.hook # Learner.summary


logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info(
    "Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures

# %% [markdown]
# Papermill script parameters:

# %% tags=["hide-input"]
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
# files and folders
# Datasplit folder with data for experiment
folder_experiment: str = 'runs/example'
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # change default to pickled files
# training
epochs_max: int = 20  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
patience: int = 1  # Patience for early stopping
batch_size: int = 32_768  # Batch size for training (and evaluation)
cuda: bool = True  # Use the GPU for training?
# model
# Dimensionality of encoding dimension (latent space of model)
latent_dim: int = 10
sample_idx_position: int = 0  # position of index which is sample ID
model: str = 'CF'  # model name
model_key: str = 'CF'  # potentially alternative key for model (grid search)
save_pred_real_na: bool = True  # Save all predictions for missing values

# %% [markdown]
# Some argument transformations

# %% tags=["hide-input"]
args = vaep.nb.get_params(args, globals=globals())
args

# %% tags=["hide-input"]
args = vaep.nb.args_from_dict(args)

# # Currently not needed -> DotProduct used, not a FNN
# if isinstance(args.hidden_layers, str):
#     args.overwrite_entry("hidden_layers", [int(x) for x in args.hidden_layers.split('_')])
# else:
#     raise ValueError(f"hidden_layers is of unknown type {type(args.hidden_layers)}")
args


# %% [markdown]
# Some naming conventions

# %% tags=["hide-input"]
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'

if not args.cuda:
    default_device(use=False)  # set to cpu


# %% [markdown]
# ## Load data in long format

# %% tags=["hide-input"]
data = datasplits.DataSplits.from_folder(
    args.data, file_format=args.file_format)

# %% [markdown]
# data is loaded in long format

# %% tags=["hide-input"]
data.train_X

# %% tags=["hide-input"]
# # ! add check that specified data is available
# silent error in fastai if e.g. target column is not available

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
# ### Use some simulated missing for evaluation

# %% [markdown]
# The validation simulated NA is used to by all models to evaluate training performance.

# %% tags=["hide-input"]
val_pred_simulated_na = data.val_y.to_frame(name='observed')
val_pred_simulated_na

# %% tags=["hide-input"]
test_pred_simulated_na = data.test_y.to_frame(name='observed')
test_pred_simulated_na.describe()


# %% [markdown]
# ## Collaborative Filtering
#
# - save custom collab batch size (increase AE batch size by a factor), could be setup separately.
# - the test data is used to evaluate the performance after training

# %% tags=["hide-input"]
# larger mini-batches speed up training
ana_collab = models.collab.CollabAnalysis(
    datasplits=data,
    sample_column=sample_id,
    item_column=index_column,  # not generic
    target_column='intensity',
    model_kwargs=dict(n_factors=args.latent_dim,
                      y_range=(int(data.train_X.min()),
                               int(data.train_X.max()) + 1)
                      ),
    batch_size=args.batch_size)

# %% tags=["hide-input"]
print("Args:")
pprint(ana_collab.model_kwargs)


# %% tags=["hide-input"]
ana_collab.model = EmbeddingDotBias.from_classes(
    classes=ana_collab.dls.classes,
    **ana_collab.model_kwargs)

args.n_params = models.calc_net_weight_count(ana_collab.model)
ana_collab.params['n_parameters'] = args.n_params
ana_collab.learn = Learner(dls=ana_collab.dls, model=ana_collab.model, loss_func=MSELossFlat(),
                           cbs=EarlyStoppingCallback(patience=args.patience),
                           model_dir=args.out_models)
if args.cuda:
    ana_collab.learn.model = ana_collab.learn.model.cuda()
else:
    # try to set explicitly cpu in case not cuda
    # MPS logic might not work properly in fastai yet https://github.com/fastai/fastai/pull/3858
    ana_collab.learn.model = ana_collab.learn.model.cpu()

# learn.summary() # see comment at DAE

# %% [markdown]
# ### Training

# %% tags=["hide-input"]
# papermill_description=train_collab
suggested_lr = ana_collab.learn.lr_find()
print(f"{suggested_lr.valley = :.5f}")
ana_collab.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)
args.epoch_trained = ana_collab.learn.epoch + 1
# ana_collab.learn.fit_one_cycle(args.epochs_max, lr_max=1e-3)
ana_collab.model_kwargs['suggested_inital_lr'] = suggested_lr.valley
ana_collab.learn.save('collab_model')
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('CF loss: Reconstruction loss')
ana_collab.learn.recorder.plot_loss(skip_start=5, ax=ax)
recorder_dump = RecorderDump(
    recorder=ana_collab.learn.recorder, name='CF')
recorder_dump.save(args.out_figures)
del recorder_dump
vaep.savefig(fig, name='collab_training',
             folder=args.out_figures)
ana_collab.model_kwargs['batch_size'] = ana_collab.batch_size
vaep.io.dump_json(ana_collab.model_kwargs, args.out_models /
                  TEMPLATE_MODEL_PARAMS.format('CF'))

# %% [markdown]
# ### Predictions

# %% [markdown]
# Compare simulated_na data predictions to original values

# %% tags=["hide-input"]
# this could be done using the validation data laoder now
ana_collab.test_dl = ana_collab.dls.test_dl(
    data.val_y.reset_index())  # test_dl is here validation data
val_pred_simulated_na['CF'], _ = ana_collab.learn.get_preds(
    dl=ana_collab.test_dl)
val_pred_simulated_na


# %% [markdown]
# select test data predictions

# %% tags=["hide-input"]
ana_collab.test_dl = ana_collab.dls.test_dl(data.test_y.reset_index())
test_pred_simulated_na['CF'], _ = ana_collab.learn.get_preds(dl=ana_collab.test_dl)
test_pred_simulated_na

# %% tags=["hide-input"]
if args.save_pred_real_na:
    pred_real_na = models.collab.get_missing_values(
        df_train_long=data.train_X,
        val_idx=data.val_y.index,
        test_idx=data.test_y.index,
        analysis_collab=ana_collab)
    pred_real_na.to_csv(args.out_preds / f"pred_real_na_{args.model_key}.csv")


# %% [markdown]
# ## Data in wide format
#
# - Autoencoder need data in wide format

# %% tags=["hide-input"]
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()


# %% [markdown]
# ### Validation data
#
# - all measured (identified, observed) peptides in validation data
#
# > Does not make to much sense to compare collab and AEs,
# > as the setup differs of training and validation data differs

# %% tags=["hide-input"]
# papermill_description=metrics
d_metrics = models.Metrics()

# %% [markdown]
# The simulated NA for the validation step are real test data (not used for training nor early stopping)

# %%
added_metrics = d_metrics.add_metrics(val_pred_simulated_na, 'valid_simulated_na')
added_metrics

# %% [markdown]
# ### Test Datasplit
#
# Simulated NAs : Artificially created NAs. Some data was sampled and set
# explicitly to misssing before it was fed to the model for
# reconstruction.

# %%
added_metrics = d_metrics.add_metrics(test_pred_simulated_na, 'test_simulated_na')
added_metrics

# %% [markdown]
# Save all metrics as json

# %%
vaep.io.dump_json(d_metrics.metrics, args.out_metrics /
                  f'metrics_{args.model_key}.json')


# %%
metrics_df = models.get_df_from_nested_dict(
    d_metrics.metrics, column_levels=['model', 'metric_name']).T
metrics_df

# %% [markdown]
# ## Save predictions

# %% tags=["hide-input"]
# save simulated missing values for both splits
val_pred_simulated_na.to_csv(args.out_preds / f"pred_val_{args.model_key}.csv")
test_pred_simulated_na.to_csv(args.out_preds / f"pred_test_{args.model_key}.csv")

# %% [markdown]
# ## Config

# %% tags=["hide-input"]
args.dump(fname=args.out_models / f"model_config_{args.model_key}.yaml")
args

# %% tags=["hide-input"]
