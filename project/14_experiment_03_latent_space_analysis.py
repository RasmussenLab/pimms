# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Latent space visualization

# %%
import logging
from pathlib import Path
from pprint import pprint
from typing import Union, List
from src.nb_imports import *
import plotly.express as px

from typing import Union

from fastai.losses import MSELossFlat
from fastai.learner import Learner


import fastai
# from fastai.tabular.all import *

from fastai.basics import *
from fastai.callback.all import *
from fastai.torch_basics import *
from fastai.data.all import *

from fastai.tabular.all import *
from fastai.collab import *

# overwriting Recorder callback with custom plot_loss
from vaep.models import plot_loss, RecorderDump, calc_net_weight_count
from fastai import learner
learner.Recorder.plot_loss = plot_loss
# import fastai.callback.hook # Learner.summary

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import vaep
import vaep.model
import vaep.models as models
from vaep.models import ae
from vaep.models import collab as vaep_collab
from vaep.io.datasets import DatasetWithTarget
from vaep.transform import VaepPipeline
from vaep.io import datasplits
# from vaep.io.dataloaders import get_dls, get_test_dl
from vaep import sampling

import src
from vaep.analyzers import analyzers
from src import config
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info("Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures

# %% [markdown]
# Papermill script parameters:

# %% tags=["parameters"]
# folders
folder_experiment:str = 'runs/experiment_03/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070' # Datasplit folder with data for experiment
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# training
# n_training_samples_max:int = 1000 # Maximum number of training samples to use for training. Take most recent
epochs_max:int = 10  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
batch_size:int = 64 # Batch size for training (and evaluation)
cuda:bool=True # Use the GPU for training?
# model
latent_dim:int = 10 # Dimensionality of encoding dimension (latent space of model)
hidden_layers:Union[int,str] = 3 # A space separated string of layers, '50 20' for the encoder, reverse will be use for decoder
force_train:bool = True # Force training when saved model could be used. Per default re-train model
sample_idx_position: int = 0 # position of index which is sample ID

# %% [markdown]
# Some argument transformations

# %%
args = config.Config()
args.fn_rawfile_metadata = fn_rawfile_metadata; del fn_rawfile_metadata
args.folder_experiment = Path(folder_experiment); del folder_experiment; args.folder_experiment.mkdir(exist_ok=True, parents=True)
args.out_folder = args.folder_experiment
args.data = args.folder_experiment / 'data'
args.out_figures = args.folder_experiment / 'figures'; args.out_figures.mkdir(exist_ok=True)
args.out_metrics = args.folder_experiment / 'metrics'; args.out_metrics.mkdir(exist_ok=True)
args.out_models = args.folder_experiment / 'models' ; args.out_models.mkdir(exist_ok=True)
# args.n_training_samples_max = n_training_samples_max; del n_training_samples_max
args.epochs_max = epochs_max; del epochs_max
args.batch_size = batch_size; del batch_size
args.cuda = cuda; del cuda
args.latent_dim = latent_dim; del latent_dim
args.force_train = force_train; del force_train

print(hidden_layers)
if isinstance(hidden_layers, int):
    args.hidden_layers = hidden_layers
elif isinstance(hidden_layers, str):
    args.hidden_layers = [int(x) for x in hidden_layers.split()]
                         # list(map(int, hidden_layers.split()))
else:
    raise ValueError(f"hidden_layers is of unknown type {type(hidden_layers)}")
del hidden_layers
args

# %% [markdown]
# Some naming conventions

# %%
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'

# %% [markdown]
# ## Load data in long format

# %%
data = datasplits.DataSplits.from_folder(args.data, file_format=file_format) 
# select max_train_samples

# %% [markdown]
# - data representation not to easy yet
# - should validation and test y (the imputed cases using replicates) be only generated in an application to 
#   keep unmanipulated data separate from imputed values?

# %%
# data # uncommet to see current representation

# %% [markdown]
# data is loaded in long format

# %%
data.train_X.sample(5)

# %% [markdown]
# Infer index names from long format 

# %%
index_columns = list(data.train_X.index.names)
sample_id = index_columns.pop(sample_idx_position)
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
# meta data for splits

# %%
df_meta = pd.read_csv(args.fn_rawfile_metadata, index_col=0)
df_meta.loc[data.train_X.index.levels[0]]

# %% [markdown]
# ## Initialize Comparison
#
# - replicates idea for truely missing values: Define truth as by using n=3 replicates to impute
#   each sample
# - real test data:
#     - Not used for predictions or early stopping.
#     - [x] add some additional NAs based on distribution of data

# %%
freq_peptides = sampling.frequency_by_index(data.train_X, 0)
freq_peptides.head() # training data

# %% [markdown]
# ### Produce some addional fake samples

# %% [markdown]
# The validation fake NA could be used to by all models (although the collab model will have another "validation" data split

# %%
val_pred_fake_na = data.val_y.to_frame(name='observed')
# val_pred_fake_na['interpolated'] = data.interpolate('val_X')
val_pred_fake_na

# %%
test_pred_fake_na = data.test_y.to_frame(name='observed')
# test_pred_fake_na['interpolated'] = data.interpolate('test_X') # "gold standard"
test_pred_fake_na.describe()

# %% [markdown]
# Real NA with all missing values or only with interpolated missing values
#
# - real NAs cannot be evaluated exactly. 
# - One could use a gold standard (e.g. interpolation or MBRs) to do this
# - for now the interpolation is not exact enough

# %%
# test_pred_real_na = data.interpolate('test_X').to_frame('interpolated')
# test_pred_real_na = test_pred_real_na.loc[test_pred_real_na.index.difference(test_pred_fake_na.index)]

# test_pred_real_na # does not contain NAs

# %%
# test_pred_observed = data.test_X.to_frame('measured')
# test_pred_observed.sort_index(inplace=True)

# %% [markdown]
# And predictions on validation (to see if the test data performs worse than the validation data, which was only used for early stopping)
# - possibility to also mask some predictions for model

# %%
# valid_pred = data.val_X.to_frame('measured')
# valid_pred

# %% [markdown]
# ### PCA plot of training data
#
#  > moved to data selection notebook (`14_experiment_03_data.ipynb`)

# %% [markdown]
# ## Collaborative Filtering
#
# - save custom collab batch size (increase AE batch size by a factor), could be setup separately.

# %%
# larger mini-batches speed up training
args.batch_size_collab = args.batch_size*64

ana_collab = models.collab.CollabAnalysis(datasplits=data,
                                          sample_column=sample_id,
                                          item_column=index_column, # not generic
                                          target_column='intensity',
                                          model_kwargs=dict(n_factors=args.latent_dim,
                                                            y_range=(int(data.train_X.min()),
                                                                     int(data.train_X.max())+1)
                                                            ),
                                          batch_size=args.batch_size_collab)

# %%
print("Args:")
pprint(ana_collab.model_kwargs)

# %%
ana_collab.model = EmbeddingDotBias.from_classes(
    classes=ana_collab.dls.classes,
    **ana_collab.model_kwargs)

args.n_params_collab = models.calc_net_weight_count(ana_collab.model)
ana_collab.params['n_parameters'] = args.n_params_collab
ana_collab.learn = Learner(dls=ana_collab.dls, model=ana_collab.model, loss_func=MSELossFlat(),
                cbs=EarlyStoppingCallback(patience=1),
                model_dir=args.out_models)
if args.cuda:
    ana_collab.learn.model = ana_collab.learn.model.cuda()
# learn.summary() # see comment at DAE

# %% [markdown]
# ### Training

# %%
#papermill_description=train_collab
try:
    if args.force_train:
        raise FileNotFoundError
    ana_collab.learn = ana_collab.learn.load('collab_model')
    logger.info("Loaded saved model")
    recorder_loaded = RecorderDump.load(args.out_figures, "collab")
    logger.info("Loaded dumped figure data.")
    recorder_loaded.plot_loss()
    del recorder_loaded
except FileNotFoundError:
    suggested_lr = ana_collab.learn.lr_find()
    print(f"{suggested_lr.valley = :.5f}")
    ana_collab.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)
    # ana_collab.learn.fit_one_cycle(args.epochs_max, lr_max=1e-3)
    ana_collab.model_kwargs['suggested_inital_lr'] = suggested_lr.valley
    ana_collab.learn.save('collab_model')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title('Collab loss: Reconstruction loss')
    ana_collab.learn.recorder.plot_loss(skip_start=5, ax=ax)
    recorder_dump = RecorderDump(recorder=ana_collab.learn.recorder, name='collab')
    recorder_dump.save(args.out_figures)
    del recorder_dump
    vaep.savefig(fig, name='collab_training',
                            folder=args.out_figures)
    ana_collab.model_kwargs['batch_size'] = ana_collab.batch_size
    vaep.io.dump_json(ana_collab.model_kwargs, args.out_models / TEMPLATE_MODEL_PARAMS.format("collab"))

# %% [markdown] tags=[]
# ### Predictions
# - validation data for collab is a mix of peptides both from the original training and validation data set
# - compare these to the interpolated values based on the training data **as is** for the collab-model
# - comparison for collab model will therefore not be 1 to 1 comparable with the Autoencoder models on the **validation**  data split

# %%
# valid_pred_collab = ana_collab.dls.valid_ds.new(ana_collab.dls.valid_ds.all_cols).decode().items
# pred, target = ana_collab.learn.get_preds()
# valid_pred_collab['collab'] = pred.flatten().numpy()
# index_columns = [sample_id, index_column]
# valid_pred_collab = valid_pred_collab.set_index(index_columns)
# valid_pred_collab

# %%
collab_train = ana_collab.dls.train_ds.new(ana_collab.dls.train_ds.all_cols).decode().items
collab_train = collab_train.set_index(index_columns).unstack()
val_pred_fake_na['interpolated'] = vaep.pandas.interpolate(wide_df = collab_train)

# %% [markdown]
# Compare fake_na data (not used for validation) based on original training and validation data

# %%
ana_collab.test_dl = ana_collab.dls.test_dl(data.val_y.reset_index())
val_pred_fake_na['collab'], _ = ana_collab.learn.get_preds(dl=ana_collab.test_dl)
val_pred_fake_na

# %% [markdown]
# Move everything to cpu, to make sure all tensors will be compatible

# %%
ana_collab.learn.cpu()

# %% [markdown]
# For predictions on test data, the sample embedding vector has to be initialized manuelly.
# Build new embeddings for test data.
#
# - for now: mean embeddings of closest k neighbours in training and validation data
#    - [ ] Recalculate KNN here, as for collab the training and validation samples are joined
# - optionally: weight mean by distance of training samples for new samples in PCA

# %%
# # KNN with ana_collab.X
# ana_collab.K_neighbours = 2
# ana_X = analyzers.AnalyzePeptides(data=ana_collab.X.set_index([sample_id, index_column]).squeeze(), is_wide_format=False, ind_unstack=index_column)
# # this does compute it twice, maybe add optional "get_model"?
# ana_X.df_meta = df_meta
# _ = ana_X.get_PCA(n_components=ana_collab.K_neighbours)
# ana_X.PCs = ana_X.calculate_PCs(ana_X.df)

# %%
# from sklearn.neighbors import NearestNeighbors
# nn = NearestNeighbors().fit(ana_X.PCs)

# %%
# test_PCs = ana_X.calculate_PCs(data.test_X.unstack())
# d, idx = nn.kneighbors(test_PCs) # use K neighreast neighbors from training data (add validation?)
# idx = torch.from_numpy(idx)

# # # mean embeddings
# test_sample_embeddings = ana_collab.learn.u_weight(idx).mean(1)
# test_sample_biases     = ana_collab.learn.u_bias(idx).mean(1)
# # # mean corresponds to equal weights summed
# # w = np.ones(5)
# # w = w / w.sum()
# # w = np.expand_dims(w, axis=-1)


# # # mean embeddings weighted by distance

# # w = d / d.sum()
# # w = np.expand_dims(w, axis=-1)

# # test_sample_embeddings = (learn.u_weight(idx).detach() * w).sum(1)
# # test_sample_biases     = (learn.u_bias(idx).detach() * w).sum(1)

# %%
# test_pred_collab = vaep_collab.collab_prediction(idx_samples=idx,
#                                      learn=ana_collab.learn,
#                                      index_samples=test_PCs.index)
# test_pred_collab

ana_collab.test_dl = ana_collab.dls.test_dl(data.test_y.reset_index())
test_pred_fake_na['collab'], _ = ana_collab.learn.get_preds(dl=ana_collab.test_dl)
test_pred_fake_na

# %%
# test_pred_collab = test_pred_collab.stack()
# test_pred_fake_na['collab'] = test_pred_collab
# test_pred_real_na['collab'] = test_pred_collab
# test_pred_observed['collab'] = test_pred_collab

# %% [markdown]
# ## Data in wide format
#
# - Autoencoder need data in wide format

# %%
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()

# %% [markdown]
# Calculate hidden layer dimensionality based on latent space dimension and number of hidden layers requested:

# %%
if isinstance(args.hidden_layers, int):
    args.overwrite_entry(entry='hidden_layers',
                         value=ae.get_funnel_layers(dim_in=args.M, dim_latent=args.latent_dim, n_layers=args.hidden_layers))
args

# %% [markdown]
# ## Denoising Autoencoder

# %% [markdown]
# ### Analysis: DataLoaders, Model, transform

# %%
data.train_X

# %%
data.val_y = pd.DataFrame(pd.NA, index=data.train_X.index, columns=data.train_X.columns).fillna(data.val_y)

# %%
dae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])

ana_dae = ae.AutoEncoderAnalysis(#datasplits=data,
                                 train_df=data.train_X,
                                 val_df=data.val_y, # 
                                 model=ae.Autoencoder,
                                 transform=dae_default_pipeline,
                                 decode=['normalize'],
                                 model_kwargs=dict(n_features=data.train_X.shape[-1],
                                                     n_neurons=args.hidden_layers,
                                                     last_decoder_activation=None,
                                                     dim_latent=args.latent_dim),
                                 bs=args.batch_size)
args.n_params_dae = ana_dae.n_params_ae

if args.cuda:
    ana_dae.model = ana_dae.model.cuda()
ana_dae.model

# %%
# import importlib; importlib.reload(ae)

# %%
# ana_dae.dls.train_ds[0]

# %%
# t_mask, t_data, t_target = ana_dae.dls.valid_ds[1]
# t_target[t_mask != 1]

# %% [markdown]
# ### Learner

# %%
ana_dae.learn = Learner(dls=ana_dae.dls, model=ana_dae.model,
                loss_func=MSELossFlat(),
                cbs=[EarlyStoppingCallback(), ae.ModelAdapter()]
                 )

# %%
ana_dae.learn.show_training_loop()

# %% [markdown]
# Adding a `EarlyStoppingCallback` results in an error.  Potential fix in [PR3509](https://github.com/fastai/fastai/pull/3509) is not yet in current version. Try again later

# %%
# learn.summary()

# %%
suggested_lr = ana_dae.learn.lr_find()
ana_dae.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr

# %%
vaep.io.dump_json(ana_dae.params, args.out_models / TEMPLATE_MODEL_PARAMS.format("dae"))

# %% [markdown]
# ### Training
#

# %%
#papermill_description=train_dae
ana_dae.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)


# %%
def plot_training_losses(learner:fastai.learner.Learner, name:str, ax=None, save_recorder:bool=True, folder='figures', figsize=(15,8)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.set_title(f'{name} loss: Reconstruction loss')
        learner.recorder.plot_loss(skip_start=5, ax=ax)
        name = name.lower()
        _ = RecorderDump(learner.recorder, name).save(args.out_figures)
        vaep.savefig(fig, name=f'{name}_training',
                        folder=folder)
        return fig
    
fig = plot_training_losses(learner=ana_dae.learn, name='DAE', folder=args.out_figures)

# %% [markdown]
# ### Predictions
#
# - data of training data set and validation dataset to create predictions is the same.
# - predictions include real NA (which are not further compared)
#
# - validation and test dataset could be combined, but performance is more or less equal
# - [ ] double check ModelAdapter

# %%
pred, target = ana_dae.get_preds_from_df(df_wide=data.train_X) # train_X 
pred = pred.stack()
# valid_pred['DAE'] = pred.stack()
val_pred_fake_na['DAE'] = pred
val_pred_fake_na

# %% [markdown]
# - on test dataset

# %%
# res = ae.get_preds_from_df(df=data.test_X, learn=ana_dae.learn, transformer=ana_dae.transform)
# pred, target = res

# %% tags=[]
test_pred_fake_na['DAE'] = pred 
test_pred_fake_na

# %% [markdown]
# ### Plots
#
# - validation data
# - [ ] add test data

# %%
# could also be a method
ana_dae.model = ana_dae.model.cpu()
df_dae_latent = vaep.model.get_latent_space(ana_dae.model.encoder, dl=ana_dae.dls.valid, dl_index=ana_dae.dls.valid.data.index)
df_dae_latent

# %%
# val_meta = df_meta.loc[data.val_X.index]
df_meta

# %%
ana_latent_dae = analyzers.LatentAnalysis(df_dae_latent, df_meta, 'DAE', folder=args.out_figures)
figures['latent_DAE_by_date'], ax = ana_latent_dae.plot_by_date('Content Creation Date')

# %%
figures['latent_DAE_by_ms_instrument'], ax = ana_latent_dae.plot_by_category('instrument serial number')

# %% [markdown]
# ## Variational Autoencoder

# %% [markdown]
# ### Transform of data

# %%
vae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', MinMaxScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])



# %% [markdown]
# ### Analysis: DataLoaders, Model

# %%
from torch.nn import Sigmoid

ana_vae = ae.AutoEncoderAnalysis(#datasplits=data,
                                 train_df=data.train_X,
                                 val_df=data.val_y, # 
                    model=ae.VAE,
                    model_kwargs=dict(n_features=data.train_X.shape[-1],
                                      n_neurons=args.hidden_layers,
                                      last_encoder_activation=None,
                                      last_decoder_activation=Sigmoid,
                                      dim_latent=args.latent_dim),
                   transform = vae_default_pipeline,
                   decode=['normalize'])
args.n_params_vae = ana_vae.n_params_ae
if args.cuda:
    ana_vae.model = ana_vae.model.cuda()
ana_vae.model

# %% [markdown]
# ### Training

# %%
#papermill_description=train_vae
ana_vae.learn = Learner(dls=ana_vae.dls,
                model=ana_vae.model,
                loss_func=ae.loss_fct_vae,
                cbs=[ae.ModelAdapterVAE(), EarlyStoppingCallback()])

ana_vae.learn.show_training_loop()
# learn.summary() # see comment above under DAE

# %%
suggested_lr = ana_vae.learn.lr_find()
ana_vae.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr

# %% [markdown]
# dump model config

# %%
# needs class as argument, not instance, but serialization needs instance
ana_vae.params['last_decoder_activation'] = Sigmoid()

vaep.io.dump_json(
    vaep.io.parse_dict(
        ana_vae.params, types=[
            (torch.nn.modules.module.Module, lambda m: str(m))
        ]),
    args.out_models / TEMPLATE_MODEL_PARAMS.format("vae"))

# restore original value
ana_vae.params['last_decoder_activation'] = Sigmoid

# %%
ana_vae.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)

# %%
fig = plot_training_losses(ana_vae.learn, 'VAE', folder=args.out_figures)

# %% [markdown]
# ### Predictions
# validation data set

# %%
pred, target =res = ae.get_preds_from_df(df=data.train_X, learn=ana_vae.learn, 
                                         position_pred_tuple=0, 
                                         transformer=ana_vae.transform)
# valid_pred['VAE'] = pred.stack()
val_pred_fake_na['VAE'] = pred.stack()

# %% [markdown]
# test data set

# %%
# pred, target = ae.get_preds_from_df(df=data.test_X, learn=ana_vae.learn, 
#                         position_pred_tuple=0,
#                         transformer=ana_vae.transform)

# test_pred_observed['VAE'] = pred.stack()
test_pred_fake_na['VAE']  = pred.stack()
# test_pred_real_na['VAE']  = pred.stack()

# %% [markdown]
# ### Plots
#
# - validation data

# %%
ana_vae.model = ana_vae.model.cpu()
df_vae_latent = vaep.model.get_latent_space(ana_vae.model.get_mu_and_logvar, dl=ana_vae.dls.valid, dl_index=ana_vae.dls.valid.data.index)
df_vae_latent

# %%
_model_key = 'VAE'
ana_latent_vae = analyzers.LatentAnalysis(df_vae_latent, df_meta, _model_key, folder=args.out_figures)
figures[f'latent_{_model_key}_by_date'], ax = ana_latent_vae.plot_by_date('Content Creation Date')

# %%
_cat = 'ms_instrument' # Could be created in data as an ID from three instrument variables
figures[f'latent_{_model_key}_by_{_cat}'], ax = ana_latent_vae.plot_by_category('instrument serial number')

# %% [markdown]
# ## Comparisons
#
# > Note: The interpolated values have less predictions for comparisons than the ones based on models (Collab, DAE, VAE)  
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
#papermill_description=metrics
d_metrics = models.Metrics(no_na_key='NA interpolated', with_na_key='NA not interpolated')
# added_metrics = d_metrics.add_metrics(valid_pred_collab, 'valid_collab')
# pd.DataFrame(added_metrics['no_na'])

# %% tags=[]
# added_metrics = d_metrics.add_metrics(valid_pred_collab, 'valid_collab')
# added_metrics

# %% tags=[]
# test_metrics = models.get_metrics_df(pred_df = valid_pred_collab.dropna())
# assert test_metrics ==  added_metrics[d_metrics.no_na_key]

# %%
# added_metrics = d_metrics.add_metrics(valid_pred, 'valid_ae_observed')
# added_metrics

# %% [markdown]
# The fake NA for the validation step are in fact real test data (not used for training nor early stopping), but to not confuse the test-data split with these, I omit it here. 

# %% tags=[]
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics

# %% [markdown] tags=[]
# ### Test Datasplit
#
# Fake NAs : Artificially created NAs. Some data was sampled and set explicitly to misssing before it was fed to the model for reconstruction.

# %% tags=[]
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics

# %% [markdown]
# Non missing data, which was fed to the model

# %%
# added_metrics = d_metrics.add_metrics(test_pred_observed, 'test_observed')
# added_metrics

# %% [markdown]
# True NA data which was interpolated by other samples
# > Comparing to imputation with each other might not be sensible

# %%
# added_metrics = d_metrics.add_metrics(test_pred_real_na, 'test_real_na')
# added_metrics

# %%
# analysis per sample?
# analysis per peptide?
# (test_pred_real_na['interpolated'] - test_pred_real_na['DAE']).sort_values().plot(rot=45)

# %% [markdown]
# Save all metrics as json

# %% tags=[]
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / 'metrics.json')


# %% tags=[]
def get_df_from_nested_dict(nested_dict, column_levels=['data_split', 'model', 'metric_name']):
    metrics = {}
    for k, run_metrics in nested_dict.items():
        metrics[k] = vaep.pandas.flatten_dict_of_dicts(run_metrics)

    metrics_dict_multikey = metrics

    metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics.columns.names = column_levels
    metrics.index.name = 'subset'
    return metrics

metrics_df = get_df_from_nested_dict(d_metrics.metrics).T
metrics_df

# %% [markdown]
# ### Plot metrics

# %%
plotly_view = metrics_df.stack().unstack(-2).set_index('N', append=True)
plotly_view.head()

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
                 height=500
                 )
fig.show()

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
                 height=500
                 )
fig.show()

# %% [markdown] tags=[]
# ## Config

# %%
args.dump()
args
