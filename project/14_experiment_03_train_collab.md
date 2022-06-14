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

# Collaborative Filtering

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
folder_data:str = '' # specify data directory if needed
file_format: str = 'pkl' # change default to pickled files
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# training
epochs_max:int = 20  # Maximum number of epochs
# early_stopping:bool = True # Wheather to use early stopping or not
batch_size_collab:int = 32_768 # Batch size for training (and evaluation)
cuda:bool=True # Use the GPU for training?
# model
latent_dim:int = 10 # Dimensionality of encoding dimension (latent space of model)
hidden_layers:Union[int,str] = 3 # A space separated string of layers, '50 20' for the encoder, reverse will be use for decoder
force_train:bool = True # Force training when saved model could be used. Per default re-train model
sample_idx_position: int = 0 # position of index which is sample ID
```

```python
# folder_experiment = "runs/experiment_03/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070"
# folder_experiment = "runs/experiment_03/df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070"
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
args.batch_size_collab = batch_size_collab
del batch_size_collab
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
    args.hidden_layers = [int(x) for x in hidden_layers.split('_')]
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
# args.batch_size_collab = args.batch_size

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
_model_key = 'collab'
vaep.io.dump_json(d_metrics.metrics, args.out_metrics / f'metrics_{_model_key}.json')
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
val_pred_fake_na.to_csv(args.out_preds / f"pred_val_{_model_key}.csv")
test_pred_fake_na.to_csv(args.out_preds / f"pred_test_{_model_key}.csv")
```

<!-- #region tags=[] -->
## Config
<!-- #endregion -->

```python
args.dump(fname=args.out_models/ f"model_config_{_model_key}.yaml")
args
```
