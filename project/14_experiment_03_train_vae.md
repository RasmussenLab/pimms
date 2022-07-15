---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Variational Autoencoder

```python
import logging
from pathlib import Path
from pprint import pprint
from typing import Union, List

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


import vaep.nb as config
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info("Experiment 03 - Analysis of latent spaces and performance comparisions")

figures = {}  # collection of ax or figures
```

Papermill script parameters:

```python tags=["parameters"]
# files and folders
folder_experiment:str = 'runs/experiment_03/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070' # Datasplit folder with data for experiment
folder_data:str = '' # specify data directory if needed
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# training
epochs_max:int = 50  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
batch_size:int = 64 # Batch size for training (and evaluation)
cuda:bool=True # Use the GPU for training?
# model
latent_dim:int = 25 # Dimensionality of encoding dimension (latent space of model)
hidden_layers:Union[int,str] = '256_128' # A underscore separated string of layers, '128_128' for the encoder, reverse will be use for decoder
force_train:bool = True # Force training when saved model could be used. Per default re-train model
sample_idx_position: int = 0 # position of index which is sample ID
model_key = 'VAE'
```

```python
# # folder_experiment = "runs/experiment_03/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070"
# folder_experiment = "runs/experiment_03/df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070"
# latent_dim = 30
# # epochs_max = 2
# # force_train = False
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
if folder_data:
    args.data = Path(folder_data)
else:
    args.data = args.folder_experiment / 'data'
assert args.data.exists(), f"Directory not found: {args.data}"
del folder_data
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
if isinstance(hidden_layers, str):
    args.hidden_layers = [int(x) for x in hidden_layers.split('_')]
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

## Initialize Comparison

- replicates idea for truely missing values: Define truth as by using n=3 replicates to impute
  each sample
- real test data:
    - Not used for predictions or early stopping.
    - [x] add some additional NAs based on distribution of data

```python
freq_feat = sampling.frequency_by_index(data.train_X, 0)
freq_feat.name = 'freq'
freq_feat.head() # training data
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


## Data in wide format

- Autoencoder need data in wide format

```python
data.to_wide_format()
args.M = data.train_X.shape[-1]
data.train_X.head()
```


### Add interpolation performance

```python
interpolated = vaep.pandas.interpolate(wide_df = data.train_X) 
val_pred_fake_na['interpolated'] = interpolated
test_pred_fake_na['interpolated'] = interpolated
del interpolated
test_pred_fake_na
```

```python
# Add median pred performance
medians_train = data.train_X.median()
medians_train.name = 'median'

val_pred_fake_na = val_pred_fake_na.join(medians_train)
test_pred_fake_na = test_pred_fake_na.join(medians_train)
val_pred_fake_na
```

### Fill Validation data with potentially missing features

```python
data.train_X
```

```python
data.val_y # potentially has less features
```

```python tags=[]
data.val_y = pd.DataFrame(pd.NA, index=data.train_X.index, columns=data.train_X.columns).fillna(data.val_y)
data.val_y
```

## Variational Autoencoder


### Transform of data

```python
vae_default_pipeline = sklearn.pipeline.Pipeline(
    [
        ('normalize', StandardScaler()),
        ('impute', SimpleImputer(add_indicator=False))
    ])
```

### Analysis: DataLoaders, Model

```python
from torch.nn import Sigmoid

ana_vae = ae.AutoEncoderAnalysis(  # datasplits=data,
    train_df=data.train_X,
    val_df=data.val_y,
    model= models.vae.VAE,
    model_kwargs=dict(n_features=data.train_X.shape[-1],
                      h_layers=args.hidden_layers,
                          # last_encoder_activation=None,
                      last_decoder_activation=None,
                      dim_latent=args.latent_dim),
    transform=vae_default_pipeline,
    decode=['normalize'],
    bs=args.batch_size)
args.n_params_vae = ana_vae.n_params_ae
if args.cuda:
    ana_vae.model = ana_vae.model.cuda()
ana_vae.model
```

### Training

```python
#self.loss_func(self.pred, *self.yb)

results = []
loss_fct = partial(models.vae.loss_fct, results=results)
```

```python
# papermill_description=train_vae
ana_vae.learn = Learner(dls=ana_vae.dls,
                        model=ana_vae.model,
                        loss_func=loss_fct,
                        cbs=[ae.ModelAdapterVAE(),
                             EarlyStoppingCallback(patience=50)
                             ])

ana_vae.learn.show_training_loop()
# learn.summary() # see comment above under DAE
```

```python
suggested_lr = ana_vae.learn.lr_find()
ana_vae.params['suggested_inital_lr'] = suggested_lr.valley
suggested_lr
```

```python
results.clear() # reset results
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
    args.out_models / TEMPLATE_MODEL_PARAMS.format(model_key.lower()))

# restore original value
ana_vae.params['last_decoder_activation'] = Sigmoid
```

```python tags=[]
ana_vae.learn.fit_one_cycle(args.epochs_max, lr_max=suggested_lr.valley)
```

Save number of actually trained epochs

```python
args.epoch_vae = ana_vae.learn.epoch + 1
args.epoch_vae
```

```python
# results are mixed (train and validation) -> better design needed
# in general: L_rec >> L_reg (seems so)
# # rename _results!
# results_train = pd.DataFrame.from_records(_results[::2], columns=['L_rec', 'L_reg'])
# results_train.index.name = 'step'
# results_train.plot()
```

```python tags=[]
N_train_notna = data.train_X.notna().sum().sum()
N_val_notna = data.val_y.notna().sum().sum()
fig = models.plot_training_losses(ana_vae.learn, model_key, folder=args.out_figures, norm_factors=[N_train_notna, N_val_notna]) # non-normalized plot of total loss
```

### Predictions
create predictions and select validation data predictions

```python
pred, target = res = ae.get_preds_from_df(df=data.train_X, learn=ana_vae.learn,
                                          position_pred_tuple=0,
                                          transformer=ana_vae.transform)
pred
```

```python
val_pred_fake_na['VAE'] = pred.stack() # 'model_key' ?
val_pred_fake_na
```

select test data predictions

```python
test_pred_fake_na['VAE'] = pred.stack() # model_key?
test_pred_fake_na
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
ana_latent_vae = analyzers.LatentAnalysis(df_vae_latent,
                                          df_meta,
                                          model_key,
                                          folder=args.out_figures)
figures[f'latent_{model_key.lower()}_by_date'], ax = ana_latent_vae.plot_by_date(
    'Content Creation Date')
```

```python
# Could be created in data as an ID from three instrument variables
_cat = 'ms_instrument'
figures[f'latent_{model_key.lower()}_by_{_cat}'], ax = ana_latent_vae.plot_by_category('instrument serial number')
```

```python
errors_val = val_pred_fake_na.drop('observed', axis=1).sub(val_pred_fake_na['observed'], axis=0)
errors_val = errors_val.abs().groupby(level=-1).mean()
errors_val = errors_val.join(freq_feat).sort_values(by='freq', ascending=True)

errors_val_smoothed = errors_val.copy()
errors_val_smoothed[errors_val.columns[:-1]] = errors_val[errors_val.columns[:-1]].rolling(window=200, min_periods=1).mean()
ax = errors_val_smoothed.plot(x='freq', figsize=(15,10) )
# errors_val_smoothed
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

### Save all metrics as json

```python tags=[]
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / f'metrics_{model_key.lower()}.json')
d_metrics
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

<!-- #region tags=[] -->
## Config
<!-- #endregion -->

## Save predictions

```python
val_pred_fake_na.to_csv(args.out_preds / f"pred_val_{model_key.lower()}.csv")
test_pred_fake_na.to_csv(args.out_preds / f"pred_test_{model_key.lower()}.csv")
```

```python
args.dump(fname=args.out_models/ f"model_config_{model_key.lower()}.yaml")
args.model_type = 'VAE'
args.model_key = model_key
args
```

```python

```
