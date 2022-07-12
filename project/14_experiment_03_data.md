---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
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

# Experiment 03 - Data

Create data splits

```python
from typing import Union, List
from dataclasses import dataclass
import logging
from pathlib import Path
from pprint import pprint
from src.nb_imports import *
pd.options.display.max_columns = 32

import plotly.express as px

from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors
from vaep.pandas import interpolate, parse_query_expression
from vaep.io.datasplits import DataSplits
from vaep.io import thermo_raw_files
from vaep.sampling import feature_frequency, frequency_by_index, sample_data

import src
from vaep.logging import setup_logger
logger = setup_logger(logger=logging.getLogger('vaep'))
logger.info("Experiment 03 - data")

figures = {}  # collection of ax or figures
```

## Arguments

```python
# catch passed parameters
args = None
args = dict(globals()).keys()
```

```python tags=["parameters"]
FN_INTENSITIES: str =  'data/single_datasets/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# FN_PEPTIDE_FREQ: str = 'data/processed/count_all_peptides.json' # Peptide counts for all parsed files on erda (for data selection)
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# M: int = 5000 # M most common features
sample_completeness: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
select_N = None # sample a certain number of samples
min_RT_time: Union[int, float] = 120 # Minum retention time (RT) in minutes
index_col: Union[str,int] = ['Sample ID', 'Gene names'] # Can be either a string or position (typical 0 for first column)
# query expression for subsetting
# query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
column_names: List = None # Manuelly set column names
```

```python
# # peptides
# FN_INTENSITIES: str = 'data/single_datasets/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# index_col: Union[str,int] = ['Sample ID', 'peptide'] # Can be either a string or position (typical 0 for first column)
# folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
```
```python
# # # evidence
# FN_INTENSITIES: str = 'data/single_datasets/df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# index_col: Union[str,int] = ['Sample ID', 'Sequence', 'Charge'] # Can be either a string or position (typical 0 for first column)
# folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
```


```python
args = {k: v for k, v in globals().items() if k not in args and k[0] != '_'}
args
```

```python
# There must be a better way...
@dataclass
class DataConfig:
    """Documentation. Copy parameters one-to-one to a dataclass."""
    FN_INTENSITIES: str  # Samples metadata extraced from erda
    # file_ext: str # file extension
    # FN_PEPTIDE_FREQ: str # Peptide counts for all parsed files on erda (for data selection)
    fn_rawfile_metadata: str  # Machine parsed metadata from rawfile workflow
    # M: int # M most common features
    sample_completeness: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
    select_N:int = None # sample a certain number of samples
    min_RT_time: Union[int, float] = 120
    index_col: Union[
        str, int
    ] = "Sample ID"  # Can be either a string or position (typical 0 for first column)
    # query expression for subsetting
    # query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
    logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
    folder_experiment: str = 'runs/experiment_03'
    column_names: str = None # Manuelly set column names


params = DataConfig(**args) # catches if non-specified arguments were passed

params = OmegaConf.create(params.__dict__)
dict(params)
```

## Setup

```python
folder_experiment = Path(folder_experiment)
folder_experiment.mkdir(exist_ok=True, parents=True)
logger.info(f'Folder for output = {folder_experiment}')

folder_data = folder_experiment / 'data'
folder_data.mkdir(exist_ok=True)
logger.info(f'Folder for data: {folder_data = }')

folder_figures = folder_experiment / 'figures'
folder_figures.mkdir(exist_ok=True)
logger.info(f'Folder for figures: {folder_figures = }')
```

<!-- #region tags=[] -->
## Raw data
<!-- #endregion -->

process arguments

```python
logger.info(f"{FN_INTENSITIES = }")


FILE_FORMAT_TO_CONSTRUCTOR = {'csv': 'from_csv',
                              'pkl': 'from_pickle',
                              'pickle': 'from_pickle',
                              }

FILE_EXT = Path(FN_INTENSITIES).suffix[1:]
logger.info(f"File format (extension): {FILE_EXT}  (!specifies data loading function!)")
```

```python tags=[]
%%time
params.used_features = None # sorted(selected_peptides)
if isinstance(params.index_col, str) and params.used_features: params.used_features.insert(0, params.index_col)
constructor = getattr(AnalyzePeptides, FILE_FORMAT_TO_CONSTRUCTOR[FILE_EXT]) #AnalyzePeptides.from_csv 
analysis = constructor(fname=params.FN_INTENSITIES,
                                     index_col=index_col,
                                     usecols=params.used_features
                                    )

if params.column_names:
    analysis.df.columns.names = params.column_names


log_fct = getattr(np, params.logarithm)
analysis.log_transform(log_fct)
logger.info(f"{analysis = }")
analysis.df
```

In case there are multiple features for each intensity values (currenlty: peptide sequence and charge), combine the column names to a single str index.

> The Collaborative Modeling approach will need a single feature column.

```python
def join_as_str(seq):
    ret = "_".join(str(x) for x in seq)
    return ret
    
# if hasattr(analysis.df.columns, "levels"):
if isinstance(analysis.df.columns, pd.MultiIndex):
    logger.warning("combine MultiIndex columns to one feature column")
    print(analysis.df.columns[:10].map(join_as_str))
    _new_name = join_as_str(analysis.df.columns.names)
    analysis.df.columns = analysis.df.columns.map(join_as_str)
    analysis.df.columns.name = _new_name
    logger.warning(f"New name: {analysis.df.columns.names = }")
```

### Select M most common features

- if number of features should be part of experiments, the selection has to be done here
- select between `analysis.M` (number of features available) and requested number of features `params.M` 
- can be random or based on most common features from counter objects

> Ignored for now, instead select based on feature availabiltiy across samples (see below)

```python tags=[]
# potentially create freq based on DataFrame
analysis.M
```

### Sample selection

- ensure unique indices

```python
assert analysis.df.index.is_unique, "Duplicates in index"
analysis.df.sort_index(inplace=True)
```

Select samples based on completeness

```python
if isinstance(params.sample_completeness, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {params.sample_completeness}\n'
    # assumes df in wide format
    params.sample_completeness = int(analysis.df.shape[1] * params.sample_completeness)
    msg += f'This translates to a minimum number of features per sample (to be included): {params.sample_completeness}'
    print(msg)

sample_counts = analysis.df.notna().sum(axis=1) # if DataFrame

mask = sample_counts > params.sample_completeness
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
print(msg)
analysis.df = analysis.df.loc[mask]
```

```python
params.used_samples = analysis.df.index.to_list()
```

```python
ax = analysis.df.T.describe().loc['count'].hist()
_ = ax.set_title('histogram of features for all eligable samples')
```

## Machine metadata

- read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

```python
df_meta = pd.read_csv(params.fn_rawfile_metadata, index_col=0)
df_meta = df_meta.loc[analysis.df.index.to_list()] # index is sample index
date_col = 'Content Creation Date' # hard-coded date column -> potential parameter
df_meta[date_col] = pd.to_datetime(df_meta[date_col])
df_meta
```

```python
cols_instrument = thermo_raw_files.cols_instrument
df_meta.groupby(cols_instrument)[date_col].agg(['count','min','max']) 
```

```python
df_meta.describe(datetime_is_numeric=True, percentiles=np.linspace(0.05, 0.95, 10))
```

set a minimum retention time

```python
# min_RT_time = 120 # minutes
msg = f"Minimum RT time maxiumum is set to {params.min_RT_time} minutes (to exclude too short runs, which are potentially fractions)."
mask_RT = df_meta['MS max RT'] >= 120 # can be integrated into query string
msg += f" Total number of samples retained: {int(mask_RT.sum())}."
print(msg)
```

```python
df_meta = df_meta.loc[mask_RT]
df_meta = df_meta.sort_values(date_col)
```

```python
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats
```

subset with variation

```python
meta_stats.loc[:, (meta_stats.loc['unique'] > 1) |  (meta_stats.loc['std'] > 0.1)]
```

check some columns describing settings
  - quite some variation due to `MS max charge`: Is it a parameter?

```python
meta_raw_settings = [
 'Thermo Scientific instrument model',
 'instrument serial number',
 'Software Version', 
 'MS max charge',
 'mass resolution',
 'beam-type collision-induced dissociation', 
 'injection volume setting',
 'dilution factor',
]
df_meta[meta_raw_settings].drop_duplicates() # index gives first example with this combination
```

view without `MS max charge`:
  - software can be updated
  - variation by `injection volume setting` and instrument over time
    - 500ng of peptides should be injected, based on concentration of peptides this setting is adjustd to get it
  - missing `dilution factor`
  

```python
to_drop = ['MS max charge']
df_meta[meta_raw_settings].drop(to_drop, axis=1).drop_duplicates() # index gives first example with this combination
```

<!-- #region -->
- check for variation in `software Version` and `injection volume setting`


Update selection of samples based on metadata (e.g. minimal retention time)
<!-- #endregion -->

```python
def add_meta_data(analysis: AnalyzePeptides, df_meta:pd.DataFrame):
    try:
        analysis.df = analysis.df.loc[df_meta.index]
    except KeyError as e:
        logger.warning(e)
        logger.warning("Ignore missing samples in quantified samples")
        analysis.df = analysis.df.loc[analysis.df.index.intersection(df_meta.index)]

    analysis.df_meta = df_meta # ToDo: Don't have preset metadata from filename
    return analysis

analysis = add_meta_data(analysis, df_meta=df_meta)
```

## Select a subset of samples if specified (reduce the number of samples)

- for interpolation to make sense, it is best to select a consecutive number of samples:
  - take N most recent samples

```python
if select_N is not None:
    select_N = min(select_N, len(analysis.df_meta))
                   
    analysis.df_meta = analysis.df_meta.iloc[-select_N:]
    
    analysis.df = analysis.df.loc[analysis.df_meta.index].dropna(how='all', axis=1)
    ax = analysis.df.T.describe().loc['count'].hist()
    _ = ax.set_title('histogram of features for all eligable samples')
    
    # updates
    sample_counts = analysis.df.notna().sum(axis=1) # if DataFrame
```

### Interactive and Single plots


Scatter plots need to become interactive.

```python
sample_counts.name = 'identified features'
```

```python
K = 2
pcs = analysis.get_PCA(n_components=K) # should be renamed to get_PCs
pcs = pcs.iloc[:,:K].join(analysis.df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs = pcs.reset_index()
pcs
```

```python
pcs.describe(include='all').T
```

```python
fig, ax = plt.subplots(figsize=(18,10))
analyzers.seaborn_scatter(pcs[pcs_name], fig, ax, meta=pcs['Thermo Scientific instrument model'])
```

```python
fig, ax = plt.subplots(figsize=(23,10))
analyzers.plot_date_map(pcs[pcs_name], fig, ax, pcs[date_col])
vaep.savefig(fig, folder_figures / 'pca_sample_by_date')
```

- software version: Does it make a difference?
- size: number of features in a single sample

```python
fig = px.scatter(
    pcs, x=pcs_name[0], y=pcs_name[1],
    hover_name='Sample ID',
    # hover_data=analysis.df_meta,
    title=f'First two Principal Components of {analysis.M} most abundant peptides for {pcs.shape[0]} samples',
    # color=pcs['Software Version'],
    color='identified features',
    width=1200,
    height=600
)
fig.write_image(folder_figures / 'pca_identified_features.png')
fig
```

## Sample Medians and percentiles

- see boxplot [function in R](https://github.dev/symbioticMe/proBatch/blob/8fae15049be67693bd0d4a4383b51bfb4fb287a6/R/initial_assessment.R#L199-L317), which is used for [publication figure](https://github.dev/symbioticMe/batch_effects_workflow_code/blob/696eb609a55ba9ece68b732e616d7ebeaa660373/AgingMice_study/analysis_AgingMice/5b_Fig2_initial_assessment_normalization.R)
- check boxplot functions: [bokeh](https://docs.bokeh.org/en/latest/docs/gallery/boxplot.html), [plotly](https://plotly.com/python/box-plots/), [eventplot](https://matplotlib.org/stable/gallery/lines_bars_and_markers/eventplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-eventplot-demo-py)

```python
analysis.df.head()
```

```python
df = analysis.df
df = df.join(df_meta[date_col])
df = df.set_index(date_col).sort_index().to_period('min').T
df
```

```python
ax = df.boxplot(rot=80, figsize=(20, 10), fontsize='large', showfliers=False, showcaps=False)
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
vaep.savefig(fig, folder_figures / 'median_boxplot')
figures['median_boxplot'] = fig
```


Plot sample median over time
  - check if points are equally spaced (probably QC samples are run in close proximity)
  - the machine will be not use for intermediate periods

```python
dates = df_meta[date_col].sort_values()
dates.name = 'date'
median_sample_intensity = (analysis.df
                           .median(axis=1)
                           .to_frame('median intensity'))
median_sample_intensity = median_sample_intensity.join(dates)

ax = median_sample_intensity.plot.scatter(x='date', y='median intensity',
                                          rot=90,
                                          fontsize='large',
                                          figsize=(20, 10),
                                          xticks=vaep.plotting.select_dates(
                                              median_sample_intensity['date'])
                                          )
```


- the closer the labels are there denser the samples are measured aroudn that time.


## Peptide frequency  in data

- higher count, higher probability to be sampled into training data
- missing peptides are sampled both into training as well as into validation dataset
- everything not in training data is validation data

Based on unmodified training data

```python
msg = "Total number of samples in training data split: {}"
print(msg.format(len(analysis.df)))
```

```python
# # analysis.splits.to_wide_format()
# assert analysis.splits is splits, "Sanity check failed."
```

Recalculate feature frequency after selecting some samples

```python
freq_per_feature = feature_frequency(analysis.df)
freq_per_feature
```

```python
# freq_per_feature.name = 'Gene names freq' # name it differently?
freq_per_feature.to_json(folder_data / 'freq_train.json')
```

Conserning sampling with frequency weights:
  - larger weight -> higher probablility of being sampled
  - weights need to be alignable to index of original DataFrame before grouping (same index)


## Split: Train, validation and test data

- test data is in clinical language often denoted as independent validation cohort
- validation data (for model)

```python
analysis.splits = DataSplits(is_wide_format=True)
splits = analysis.splits
print(f"{analysis.splits = }")
analysis.splits.__annotations__
```

### Sample targets (Fake NAs)


Add goldstandard targets for valiation and test data
- based on same day
- same instrument


Create some target values by sampling 5% of the validation and test data.

```python
analysis.to_long_format(inplace=True)
analysis.df_long
```

```python
fake_na, splits.train_X = sample_data(analysis.df_long.squeeze(), sample_index_to_drop=0, weights=freq_per_feature, frac=0.1)
assert len(splits.train_X) > len(fake_na)
splits.val_y = fake_na.sample(frac=0.5).sort_index()
splits.test_y = fake_na.loc[fake_na.index.difference(splits.val_y.index)]
# splits
```

```python
splits.test_y
```

```python
splits.val_y
```

```python
splits.train_X
```

<!-- #region tags=[] -->
### Save in long format

- Data in long format: (peptide, sample_id, intensity)
- no missing values kept
<!-- #endregion -->

```python
splits.dump(folder=folder_data, file_format=FILE_EXT)  # dumps data in long-format
```

```python
# # Reload from disk
splits = DataSplits.from_folder(folder_data, file_format=FILE_EXT)
```

## PCA plot of training data - with filename metadata

- [ ] update: uses old metadata reading to indicate metadata derived from filename


```python
ana_train_X = analyzers.AnalyzePeptides(data=splits.train_X, is_wide_format=False, ind_unstack=splits.train_X.index.names[1:])
figures['pca_train'] = ana_train_X.plot_pca()
vaep.savefig(figures['pca_train'], folder_figures / f'pca_plot_raw_data_w_filename_meta_{ana_train_X.fname_stub}')
# ana_train_X = add_meta_data(ana_train_X) # inplace, returns ana_train_X
```

## Save parameters

```python
print(OmegaConf.to_yaml(params))
```

```python
with open(folder_experiment/'data_config.yaml', 'w') as f:
    OmegaConf.save(params, f)
```

```python

```
