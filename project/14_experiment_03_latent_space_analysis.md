---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Latent space visualization

```python
import logging
from pathlib import Path
from pprint import pprint
from typing import Union, List

from src.nb_imports import *

import plotly.express as px

# from fastai.losses import MSELossFlat
# from fastai.learner import Learner


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
from vaep.analyzers import analyzers
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
from src import config
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info("Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures
```

Papermill script parameters:

```python tags=["parameters"]
# files and folders
folder_experiment:str = 'runs/experiment_03/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070' # Datasplit folder with data for experiment
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# training
epochs_max:int = 20  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
batch_size:int = 64 # Batch size for training (and evaluation)
cuda:bool=True # Use the GPU for training?
# model
latent_dim:int = 10 # Dimensionality of encoding dimension (latent space of model)
hidden_layers:Union[int,str] = 3 # A space separated string of layers, '50 20' for the encoder, reverse will be use for decoder
force_train:bool = True # Force training when saved model could be used. Per default re-train model
sample_idx_position: int = 0 # position of index which is sample ID
```

```python
# folder_experiment = "runs/experiment_03/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070"
# latent_dim = 30
# hidden_layers = "1024 512 256" # huge input dimension
```

Some argument transformations

```python
args = config.Config()
args.fn_rawfile_metadata = fn_rawfile_metadata
del fn_rawfile_metadata
args.folder_experiment = Path(folder_experiment)
del folder_experiment
args.folder_experiment.mkdir(exist_ok=True, parents=True)
args.file_format = file_format
del file_format
args.out_folder = args.folder_experiment
args.data = args.folder_experiment / 'data'
args.out_figures = args.folder_experiment / 'figures'
args.out_figures.mkdir(exist_ok=True)
args.out_metrics = args.folder_experiment / 'metrics'
args.out_metrics.mkdir(exist_ok=True)
args.out_models = args.folder_experiment / 'models'
args.out_models.mkdir(exist_ok=True)
args.out_preds = args.folder_experiment / 'preds'
args.out_preds.mkdir(exist_ok=True)
# args.n_training_samples_max = n_training_samples_max; del n_training_samples_max
args.epochs_max = epochs_max
del epochs_max
args.batch_size = batch_size
del batch_size
args.cuda = cuda
del cuda
args.latent_dim = latent_dim
del latent_dim
args.force_train = force_train
del force_train
args.sample_idx_position = sample_idx_position
del sample_idx_position

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
```

Some naming conventions

```python
TEMPLATE_MODEL_PARAMS = 'model_params_{}.json'
```

## Load data in long format

```python
data = datasplits.DataSplits.from_folder(args.data, file_format=args.file_format) 
```

data is loaded in long format

```python
data.train_X.sample(5)
```

Infer index names from long format 

```python
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
```

load meta data for splits

```python
df_meta = pd.read_csv(args.fn_rawfile_metadata, index_col=0)
df_meta.loc[data.train_X.index.levels[0]]
```

```python
torch.cuda.current_device(), torch.cuda.memory_allocated() 
```

## Initialize Comparison

- replicates idea for truely missing values: Define truth as by using n=3 replicates to impute
  each sample
- real test data:
    - Not used for predictions or early stopping.
    - [x] add some additional NAs based on distribution of data

```python
freq_peptides = sampling.frequency_by_index(data.train_X, 0)
freq_peptides.head() # training data
```

### Produce some addional fake samples


The validation fake NA is used to by all models to evaluate training performance. 

```python
val_pred_fake_na = data.val_y.to_frame(name='observed')
val_pred_fake_na
```

```python
test_pred_fake_na = data.test_y.to_frame(name='observed')
test_pred_fake_na.describe()
```

### PCA plot of training data

 > moved to data selection notebook (`14_experiment_03_data.ipynb`)


## Collaborative Filtering

- save custom collab batch size (increase AE batch size by a factor), could be setup separately.
- the test data is used to evaluate the performance after training

```python
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
```

```python
print("Args:")
pprint(ana_collab.model_kwargs)
```

```python
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
```

### Training

```python
# papermill_description=train_collab
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
    recorder_dump = RecorderDump(
        recorder=ana_collab.learn.recorder, name='collab')
    recorder_dump.save(args.out_figures)
    del recorder_dump
    vaep.savefig(fig, name='collab_training',
                 folder=args.out_figures)
    ana_collab.model_kwargs['batch_size'] = ana_collab.batch_size
    vaep.io.dump_json(ana_collab.model_kwargs, args.out_models /
                      TEMPLATE_MODEL_PARAMS.format("collab"))
```

<!-- #region tags=[] -->
### Predictions
<!-- #endregion -->

Compare fake_na data predictions to original values

```python
# this could be done using the validation data laoder now
ana_collab.test_dl = ana_collab.dls.test_dl(data.val_y.reset_index())  # test_dl is here validation data
val_pred_fake_na['collab'], _ = ana_collab.learn.get_preds(
    dl=ana_collab.test_dl)
val_pred_fake_na
```

Move everything to cpu, to make sure all tensors will be compatible

```python
# ana_collab.learn.cpu()
```

```python
ana_collab.test_dl = ana_collab.dls.test_dl(data.test_y.reset_index())
test_pred_fake_na['collab'], _ = ana_collab.learn.get_preds(dl=ana_collab.test_dl)
test_pred_fake_na
```

free gpu memory

```python
del ana_collab
torch.cuda.current_device(), torch.cuda.memory_allocated() 
```

## Data in wide format

- Autoencoder need data in wide format

```python
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()
```

Calculate hidden layer dimensionality based on latent space dimension and number of hidden layers requested:

```python
if isinstance(args.hidden_layers, int):
    args.overwrite_entry(entry='hidden_layers',
                         value=ae.get_funnel_layers(dim_in=args.M, dim_latent=args.latent_dim, n_layers=args.hidden_layers))
args
```

### Add interpolation performance

```python
interpolated = vaep.pandas.interpolate(wide_df = data.train_X) 
val_pred_fake_na['interpolated'] = interpolated
test_pred_fake_na['interpolated'] = interpolated
del interpolated
test_pred_fake_na
```

## Denoising Autoencoder


### Analysis: DataLoaders, Model, transform

```python
data.train_X
```

```python
data.val_y = pd.DataFrame(pd.NA, index=data.train_X.index, columns=data.train_X.columns).fillna(data.val_y)
```

```python
dae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])

ana_dae = ae.AutoEncoderAnalysis(train_df=data.train_X,
                                 val_df=data.val_y,
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
```

### Learner

```python
ana_dae.learn = Learner(dls=ana_dae.dls, model=ana_dae.model,
                        loss_func=MSELossFlat(),
                        cbs=[EarlyStoppingCallback(), ae.ModelAdapter()]
                        )
```

```python
ana_dae.learn.show_training_loop()
```

Adding a `EarlyStoppingCallback` results in an error.  Potential fix in [PR3509](https://github.com/fastai/fastai/pull/3509) is not yet in current version. Try again later

```python
# learn.summary()
```

```python
suggested_lr = ana_dae.learn.lr_find()
ana_dae.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr
```

```python
vaep.io.dump_json(ana_dae.params, args.out_models / TEMPLATE_MODEL_PARAMS.format("dae"))
```

### Training


```python
# papermill_description=train_dae
ana_dae.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)
```

```python
def plot_training_losses(learner: fastai.learner.Learner, name: str, ax=None, save_recorder: bool = True, folder='figures', figsize=(15, 8)):
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
```

### Predictions

- data of training data set and validation dataset to create predictions is the same as training data.
- predictions include real NA (which are not further compared)

- [ ] double check ModelAdapter

create predictiona and select for validation data

```python
pred, target = ana_dae.get_preds_from_df(df_wide=data.train_X)  # train_X
pred = pred.stack()
val_pred_fake_na['DAE'] = pred
val_pred_fake_na
```

select predictions for test dataset

```python tags=[]
test_pred_fake_na['DAE'] = pred
test_pred_fake_na
```

### Plots

- validation data
- [ ] add test data

```python
# could also be a method
ana_dae.model = ana_dae.model.cpu()
df_dae_latent = vaep.model.get_latent_space(ana_dae.model.encoder,
                                            dl=ana_dae.dls.valid,
                                            dl_index=ana_dae.dls.valid.data.index)
df_dae_latent
```

```python
df_meta
```

```python
ana_latent_dae = analyzers.LatentAnalysis(df_dae_latent, df_meta, 'DAE', folder=args.out_figures)
figures['latent_DAE_by_date'], ax = ana_latent_dae.plot_by_date('Content Creation Date')
```

```python
figures['latent_DAE_by_ms_instrument'], ax = ana_latent_dae.plot_by_category('instrument serial number')
```

free gpu memory

```python
del ana_dae, ana_latent_dae
msg = f"device ID: {torch.cuda.current_device()} - Mem: {torch.cuda.memory_allocated():,d} bytes, {torch.cuda.memory_allocated()//1024**2:,d} MB"
print(msg)
```

## Variational Autoencoder


### Transform of data

```python
vae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', MinMaxScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])


```

### Analysis: DataLoaders, Model

```python
from torch.nn import Sigmoid

ana_vae = ae.AutoEncoderAnalysis(  # datasplits=data,
    train_df=data.train_X,
    val_df=data.val_y,
    model=ae.VAE,
    model_kwargs=dict(n_features=data.train_X.shape[-1],
                      n_neurons=args.hidden_layers,
                      last_encoder_activation=None,
                      last_decoder_activation=Sigmoid,
                      dim_latent=args.latent_dim),
    transform=vae_default_pipeline,
    decode=['normalize'])
args.n_params_vae = ana_vae.n_params_ae
if args.cuda:
    ana_vae.model = ana_vae.model.cuda()
ana_vae.model
```

### Training

```python
# papermill_description=train_vae
ana_vae.learn = Learner(dls=ana_vae.dls,
                        model=ana_vae.model,
                        loss_func=ae.loss_fct_vae,
                        cbs=[ae.ModelAdapterVAE(), EarlyStoppingCallback()])

ana_vae.learn.show_training_loop()
# learn.summary() # see comment above under DAE
```

```python
suggested_lr = ana_vae.learn.lr_find()
ana_vae.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr
```

dump model config

```python
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
```

```python tags=[]
ana_vae.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)
```

```python tags=[]
fig = plot_training_losses(ana_vae.learn, 'VAE', folder=args.out_figures)
```

### Predictions
create predictions and select validation data predictions

```python
pred, target = res = ae.get_preds_from_df(df=data.train_X, learn=ana_vae.learn,
                                          position_pred_tuple=0,
                                          transformer=ana_vae.transform)
val_pred_fake_na['VAE'] = pred.stack()
```

select test data predictions

```python
test_pred_fake_na['VAE'] = pred.stack()
```

### Plots

- validation data

```python
ana_vae.model = ana_vae.model.cpu()
df_vae_latent = vaep.model.get_latent_space(ana_vae.model.get_mu_and_logvar,
                                            dl=ana_vae.dls.valid,
                                            dl_index=ana_vae.dls.valid.data.index)
df_vae_latent
```

```python
_model_key = 'VAE'
ana_latent_vae = analyzers.LatentAnalysis(df_vae_latent,
                                          df_meta,
                                          _model_key,
                                          folder=args.out_figures)
figures[f'latent_{_model_key}_by_date'], ax = ana_latent_vae.plot_by_date(
    'Content Creation Date')
```

```python
# Could be created in data as an ID from three instrument variables
_cat = 'ms_instrument'
figures[f'latent_{_model_key}_by_{_cat}'], ax = ana_latent_vae.plot_by_category('instrument serial number')
```

## Comparisons

> Note: The interpolated values have less predictions for comparisons than the ones based on models (Collab, DAE, VAE)  
> The comparison is therefore not 100% fair as the interpolated samples will have more common ones (especailly the sparser the data)  
> Could be changed.


### Validation data

- all measured (identified, observed) peptides in validation data

> Does not make to much sense to compare collab and AEs,  
> as the setup differs of training and validation data differs

```python
# papermill_description=metrics
d_metrics = models.Metrics(no_na_key='NA interpolated', with_na_key='NA not interpolated')
```

The fake NA for the validation step are real test data (not used for training nor early stopping)

```python tags=[]
added_metrics = d_metrics.add_metrics(val_pred_fake_na, 'valid_fake_na')
added_metrics
```

<!-- #region tags=[] -->
### Test Datasplit

Fake NAs : Artificially created NAs. Some data was sampled and set explicitly to misssing before it was fed to the model for reconstruction.
<!-- #endregion -->

```python tags=[]
added_metrics = d_metrics.add_metrics(test_pred_fake_na, 'test_fake_na')
added_metrics
```

Save all metrics as json

```python tags=[]
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / 'metrics.json')
```

```python tags=[]
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
```

### Plot metrics

```python
plotly_view = metrics_df.stack().unstack(-2).set_index('N', append=True)
plotly_view.head()
```

#### Fake NA which could be interpolated

- bulk of validation and test data

```python
plotly_view.loc[pd.IndexSlice[:, :, 'NA interpolated']]
```

```python
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
```

#### Fake NA which could not be interpolated

- small fraction of total validation and test data

> not interpolated fake NA values are harder to predict for models  
> Note however: fewer predicitons might mean more variability of results

```python
plotly_view.loc[pd.IndexSlice[:, :, 'NA not interpolated']]
```

```python
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
```

## Save predictions

```python
val_pred_fake_na.to_csv(args.out_preds / f"pred_val.csv")
test_pred_fake_na.to_csv(args.out_preds / f"pred_test.csv")
```

<!-- #region tags=[] -->
## Config
<!-- #endregion -->

```python
args.dump()
args
```
