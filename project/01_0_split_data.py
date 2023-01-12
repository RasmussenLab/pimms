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
# # Experiment 03 - Data
#
# Create data splits

# %%
from typing import Union, List
from dataclasses import dataclass
import logging
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = 32

import plotly.express as px

from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors

import vaep
from vaep.pandas import interpolate, parse_query_expression
from vaep.io.datasplits import DataSplits
from vaep.io import thermo_raw_files
from vaep.sampling import feature_frequency, frequency_by_index, sample_data

from vaep.analyzers import analyzers
from vaep.analyzers.analyzers import  AnalyzePeptides

from vaep.logging import setup_logger
logger = vaep.logging.setup_nb_logger()
logger.info("Split data and make diagnostic plots")

figures = {}  # collection of ax or figures

# %% [markdown]
# ## Arguments

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
FN_INTENSITIES: str =  'data/single_datasets/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# FN_PEPTIDE_FREQ: str = 'data/processed/count_all_peptides.json' # Peptide counts for all parsed files on erda (for data selection)
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# M: int = 5000 # M most common features
feat_prevalence: Union[int, float] = 0.25 # Minum number or fraction of feature prevalence across samples to be kept
sample_completeness: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
select_N = None # sample a certain number of samples
min_RT_time: Union[int, float] = 120 # Minum retention time (RT) in minutes
index_col: Union[str,int] = ['Sample ID', 'Gene names'] # Can be either a string or position (typical 0 for first column)
# query expression for subsetting
# query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
column_names: List = None # Manuelly set column names
# metadata -> defaults for metadata extracted from machine data
meta_date_col = 'Content Creation Date'
meta_cat_col = 'Thermo Scientific instrument model'

# %%
# select_N = 50
# fn_rawfile_metadata = ''
# meta_date_col = ''
# meta_cat_col = ''
# min_RT_time = ''

# %%
# # peptides
# FN_INTENSITIES: str = 'data/single_datasets/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# index_col: Union[str,int] = ['Sample ID', 'peptide'] # Can be either a string or position (typical 0 for first column)
# folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
# %%
# # # evidence
# FN_INTENSITIES: str = 'data/single_datasets/df_intensities_evidence_long_2017_2018_2019_2020_N05015_M49321/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6075.pkl'  # Intensities for feature
# index_col: Union[str,int] = ['Sample ID', 'Sequence', 'Charge'] # Can be either a string or position (typical 0 for first column)
# folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'


# %%
args = {k: v for k, v in globals().items() if k not in args and k[0] != '_'}
args


# %%
# There must be a better way...
@dataclass
class DataConfig:
    """Documentation. Copy parameters one-to-one to a dataclass."""
    FN_INTENSITIES: str  # Samples metadata extraced from erda
    # file_ext: str # file extension
    # FN_PEPTIDE_FREQ: str # Peptide counts for all parsed files on erda (for data selection)
    fn_rawfile_metadata: str  # Machine parsed metadata from rawfile workflow
    # M: int # M most common features
    feat_prevalence: Union[int, float] = 0.25 # Minum number or fraction of feature prevalence across samples to be kept
    sample_completeness: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
    select_N:int = None # sample a certain number of samples
    min_RT_time: Union[int, float] = None # set to None per default
    index_col: Union[
        str, int
    ] = "Sample ID"  # Can be either a string or position (typical 0 for first column)
    # query expression for subsetting
    # query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
    logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
    folder_experiment: str = 'runs/experiment_03'
    column_names: str = None # Manuelly set column names
    # metadata -> defaults for metadata extracted from machine data
    meta_date_col: str = None
    meta_cat_col: str = None

params = DataConfig(**args) # catches if non-specified arguments were passed

params = OmegaConf.create(params.__dict__)
dict(params)

# %% [markdown]
# ## Setup

# %%
folder_experiment = Path(folder_experiment)
folder_experiment.mkdir(exist_ok=True, parents=True)
logger.info(f'Folder for output = {folder_experiment}')

folder_data = folder_experiment / 'data'
folder_data.mkdir(exist_ok=True)
logger.info(f'Folder for data: {folder_data = }')

folder_figures = folder_experiment / 'figures'
folder_figures.mkdir(exist_ok=True)
logger.info(f'Folder for figures: {folder_figures = }')

# %% [markdown] tags=[]
# ## Raw data

# %% [markdown]
# process arguments

# %%
logger.info(f"{FN_INTENSITIES = }")


FILE_FORMAT_TO_CONSTRUCTOR = {'csv': 'from_csv',
                              'pkl': 'from_pickle',
                              'pickle': 'from_pickle',
                              }

FILE_EXT = Path(FN_INTENSITIES).suffix[1:]
logger.info(f"File format (extension): {FILE_EXT}  (!specifies data loading function!)")

# %% tags=[]
# %%time
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

# %%
ax = analysis.df.notna().sum(axis=0).to_frame(analysis.df.columns.name).plot.box()
ax.set_ylabel('number of observation across samples')


# %% [markdown]
# In case there are multiple features for each intensity values (currenlty: peptide sequence and charge), combine the column names to a single str index.
#
# > The Collaborative Modeling approach will need a single feature column.

# %%
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

# %% [markdown] tags=[]
# ## Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %%
if params.fn_rawfile_metadata:
    df_meta = pd.read_csv(params.fn_rawfile_metadata, index_col=0)
else:
    logger.warning(f"No metadata for samples provided, create placeholder.")
    if params.meta_date_col:
        raise ValueError(f"No metadata provided, but data column set: {params.meta_date_col}")
    if params.meta_cat_col:
        raise ValueError(f"No metadata provided, but data column set: {params.meta_cat_col}")
    df_meta = pd.DataFrame(index=analysis.df.index)
df_meta = df_meta.loc[analysis.df.index.to_list()] # index is sample index
if df_meta.index.name is None:
    df_meta.index.name = params.index_col[0]
df_meta

# %%
if params.meta_date_col:
    df_meta[params.meta_date_col] = pd.to_datetime(df_meta[params.meta_date_col])
else:
    params.meta_date_col = 'PlaceholderTime'
    df_meta[params.meta_date_col] = range(len(df_meta))
df_meta

# %%
if df_meta.columns.isin(thermo_raw_files.cols_instrument).sum() == len(thermo_raw_files.cols_instrument): 
    display(df_meta.groupby(thermo_raw_files.cols_instrument)[params.meta_date_col].agg(['count','min','max']))
else:
    logger.info(f"Instrument column not found: {thermo_raw_files.cols_instrument}")

# %%
df_meta.describe(datetime_is_numeric=True, percentiles=np.linspace(0.05, 0.95, 10))

# %% [markdown]
# select samples with a minimum retention time

# %%
if params.min_RT_time:
    logger.info("Metadata should have 'MS max RT' entry from ThermoRawFileParser")
    msg = f"Minimum RT time maxiumum is set to {params.min_RT_time} minutes (to exclude too short runs, which are potentially fractions)."
    mask_RT = df_meta['MS max RT'] >= params.min_RT_time # can be integrated into query string
    msg += f" Total number of samples retained: {int(mask_RT.sum())}"
    msg += f" ({int(len(mask_RT) - mask_RT.sum())} excluded)."
    logger.info(msg)
    df_meta = df_meta.loc[mask_RT]
else:
    logger.warning(f"Retention time filtering deactivated.")

# %%
df_meta = df_meta.sort_values(params.meta_date_col)

# %%
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats

# %% [markdown]
# subset with variation

# %%
try:
    display(meta_stats.loc[:, (meta_stats.loc['unique'] > 1) |  (meta_stats.loc['std'] > 0.1)])
except KeyError:
    if 'std' in meta_stats.index:
        display(meta_stats.loc[:, (meta_stats.loc['std'] > 0.1)])
    if 'unique' in meta_stats.index:
        display(meta_stats.loc[:, (meta_stats.loc['std'] > 0.1)])

# %% [markdown]
# Optional, if using ThermoRawFileParser: check some columns describing settings
#   - software can be updated: `Software Version`
#   - `mass resolution` setting for instrument
#   - colision type for MS2: `beam-type collision-induced dissocation`
#   - missing `dilution factor` 
#   - omit (uncomment if needed):
#     - quite some variation due to `MS max charge`: omit
#     - variation by `injection volume setting` and instrument over time
#         - 500ng of peptides should be injected, based on concentration of peptides this setting is adjusted to get it

# %%
meta_raw_settings = [
 'Thermo Scientific instrument model',
 'instrument serial number',
 'Software Version', 
 # 'MS max charge',
 'mass resolution',
 'beam-type collision-induced dissociation', 
 # 'injection volume setting',
 'dilution factor',
]

if df_meta.columns.isin(meta_raw_settings).sum() == len(meta_raw_settings):
    display(
        df_meta[meta_raw_settings].drop_duplicates() # index gives first example with this combination
    )


# %% [markdown]
# - check for variation in `software Version` and `injection volume setting`
#
#
# Update selection of samples based on metadata (e.g. minimal retention time)
# - sort data the same as sorted meta data

# %%
def add_meta_data(analysis: AnalyzePeptides, df_meta:pd.DataFrame):
    try:
        analysis.df = analysis.df.loc[df_meta.index]
    except KeyError as e:
        logger.warning(e)
        logger.warning("Ignore missing samples in quantified samples")
        analysis.df = analysis.df.loc[analysis.df.index.intersection(df_meta.index)]

    analysis.df_meta = df_meta
    return analysis

analysis = add_meta_data(analysis, df_meta=df_meta)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# Ensure unique indices

# %%
assert analysis.df.index.is_unique, "Duplicates in index"

# %% [markdown]
# ## Select a subset of samples if specified (reduce the number of samples)
#
# - select features if `select_N` is specifed (for now)
# - for interpolation to make sense, it is best to select a consecutive number of samples:
#   - take N most recent samples (-> check that this makes sense for your case)

# %%
if params.select_N is not None:
    params.select_N = min(params.select_N, len(analysis.df_meta))
                   
    analysis.df_meta = analysis.df_meta.iloc[-params.select_N:]
    
    analysis.df = analysis.df.loc[analysis.df_meta.index].dropna(how='all', axis=1)
    ax = analysis.df.T.describe().loc['count'].hist()
    _ = ax.set_title('histogram of features for all eligable samples')
    
    # updates
    sample_counts = analysis.df.notna().sum(axis=1) # if DataFrame

# %%
# export Pathname captured by ThermoRawFileParser
# analysis.df_meta['Pathname'].to_json(folder_experiment / 'config_rawfile_paths.json', indent=4)

# %% [markdown]
# ## First Step: Select features by prevalence
# - `feat_prevalence` across samples

# %% tags=[]
freq_per_feature = analysis.df.notna().sum() # on wide format
if isinstance(params.feat_prevalence, float):
    N_samples = len(analysis.df_meta)
    logger.info(f"Current number of samples: {N_samples}")
    logger.info(f"Feature has to be present in at least {params.feat_prevalence:.2%} of samples")
    params.feat_prevalence = int(N_samples * params.feat_prevalence)
assert isinstance(params.feat_prevalence, int)
logger.info(f"Feature has to be present in at least {params.feat_prevalence} of samples")                
# select features
mask = freq_per_feature >= params.feat_prevalence
logger.info(f"Drop {(~mask).sum()} features")
freq_per_feature = freq_per_feature.loc[mask]
analysis.df = analysis.df.loc[:, mask]
analysis.N, analysis.M = analysis.df.shape

# # potentially create freq based on DataFrame
analysis.df

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Second step - Sample selection

# %% [markdown]
# Select samples based on completeness

# %%
if isinstance(params.sample_completeness, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {params.sample_completeness}\n'
    # assumes df in wide format
    params.sample_completeness = int(analysis.df.shape[1] * params.sample_completeness)
    msg += f'This translates to a minimum number of features per sample (to be included): {params.sample_completeness}'
    logger.info(msg)

sample_counts = analysis.df.notna().sum(axis=1) # if DataFrame
    
mask = sample_counts > params.sample_completeness
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
print(msg)
analysis.df = analysis.df.loc[mask]
analysis.df = analysis.df.dropna(axis=1, how='all') # drop now missing features

# %%
params.N, params.M = analysis.df.shape # save data dimensions
params.used_samples = analysis.df.index.to_list()

# %% [markdown]
# ### Histogram of features per sample

# %%
ax = analysis.df.notna().sum(axis=1).hist()
ax.set_xlabel('features per eligable sample')
ax.set_ylabel('observations')
fname = folder_figures / 'hist_features_per_sample'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = analysis.df.notna().sum(axis=0).sort_values().plot()
ax.set_xlabel('feature prevalence')
ax.set_ylabel('observations')
fname = folder_figures / 'feature_prevalence'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ### Number off observations accross feature value

# %%
def min_max(s: pd.Series):
    min_bin, max_bin = (int(s.min()), (int(s.max())+1))
    return min_bin, max_bin


def plot_histogram_intensites(s: pd.Series, interval_bins=1, min_max=(15, 40), ax=None):

    min_bin, max_bin = min_max
    bins = range(min_bin, int(max_bin), 1)
    ax = s.plot.hist(bins=bins, ax=ax)
    return ax, bins


min_intensity, max_intensity = min_max(analysis.df.stack())
ax, bins = plot_histogram_intensites(
    analysis.df.stack(), min_max=(min_intensity, max_intensity))
ax.locator_params(axis='x', integer=True)

fname = folder_figures / 'intensity_distribution_overall'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
missing_by_median = {'median feat value': analysis.df.median(
), 'prop. missing': analysis.df.isna().mean()}
missing_by_median = pd.DataFrame(missing_by_median)
x_col, y_col = missing_by_median.columns

bins = range(*min_max(missing_by_median['median feat value']), 1)

missing_by_median['bins'] = pd.cut(
    missing_by_median['median feat value'], bins=bins)
missing_by_median['median feat value (rounded)'] = missing_by_median['median feat value'].round(decimals=0).astype(int)
_counts = missing_by_median.groupby('median feat value (rounded)')['median feat value'].count().rename('count')
missing_by_median = missing_by_median.join(_counts, on='median feat value (rounded)')
missing_by_median['Intensity rounded (based on N observations)'] = missing_by_median.iloc[:,-2:].apply(lambda s: "{}  (N={:3,d})".format(*s), axis=1)

ax = missing_by_median.plot.scatter(x_col, y_col, ylim=(0, 1))


fname = folder_figures / 'intensity_median_vs_prop_missing_scatter'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
y_col = 'prop. missing'
x_col = 'Intensity rounded (based on N observations)'
ax = missing_by_median[[x_col, y_col]].plot.box(by=x_col)
ax = ax[0] # returned series due to by argument?
_ = ax.set_title('')
_ = ax.set_ylabel(y_col)
_ = ax.set_xlabel(x_col)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')

fname = folder_figures / 'intensity_median_vs_prop_missing_boxplot'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ### Interactive and Single plots

# %% [markdown]
# Scatter plots need to become interactive.

# %%
sample_counts.name = 'identified features'

# %%
K = 2
analysis.df = analysis.df.astype(float)
pcs = analysis.get_PCA(n_components=K) # should be renamed to get_PCs
pcs = pcs.iloc[:,:K].join(analysis.df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs = pcs.reset_index()
pcs

# %%
pcs.describe(include='all', datetime_is_numeric=True).T

# %%
if params.meta_cat_col:
    fig, ax = plt.subplots(figsize=(18,10))
    analyzers.seaborn_scatter(pcs[pcs_name], fig, ax, meta=pcs[params.meta_cat_col], title=f"by {params.meta_cat_col}")
    fname = folder_figures / f'pca_sample_by_{"_".join(params.meta_cat_col.split())}'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %%
if params.meta_date_col != 'PlaceholderTime':
    fig, ax = plt.subplots(figsize=(23, 10))
    analyzers.plot_date_map(pcs[pcs_name], fig, ax, pcs[params.meta_date_col], title=f'by {params.meta_date_col}')
    fname = folder_figures / 'pca_sample_by_date'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
# - software version: Does it make a difference?
# - size: number of features in a single sample

# %%
fig = px.scatter(
    pcs, x=pcs_name[0], y=pcs_name[1],
    hover_name=params.index_col[0],
    # hover_data=analysis.df_meta,
    title=f'First two Principal Components of {analysis.M} most abundant peptides for {pcs.shape[0]} samples',
    # color=pcs['Software Version'],
    color='identified features',
    width=1200,
    height=600
)
fname = folder_figures / 'pca_identified_features.png'
figures[fname.stem] =  fname
fig.write_image(fname)
fig # stays interactive in html

# %% [markdown]
# ## Sample Medians and percentiles

# %%
analysis.df.head()

# %%
df = analysis.df
df = df.join(df_meta[params.meta_date_col])
df = df.set_index(params.meta_date_col).sort_index()
if not params.meta_date_col == 'PlaceholderTime':
    df.to_period('min')
df = df.T

# %%
ax = df.boxplot(rot=80, figsize=(20, 10), fontsize='large', showfliers=False, showcaps=False)
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
fname = folder_figures / 'median_boxplot'
figures[fname.stem] =  fname
vaep.savefig(fig, fname)


# %% [markdown]
# Percentiles of intensities in dataset

# %%
df.stack().describe(percentiles=np.linspace(0.05, 0.95, 10))

# %% [markdown]
# ### Plot sample median over time
#   - check if points are equally spaced (probably QC samples are run in close proximity)
#   - the machine will be not use for intermediate periods

# %%
if not params.meta_date_col == 'PlaceholderTime':
    dates = df_meta[params.meta_date_col].sort_values()
    # dates.name = 'date'
    median_sample_intensity = (analysis.df
                               .median(axis=1)
                               .to_frame('median intensity'))
    median_sample_intensity = median_sample_intensity.join(dates)

    ax = median_sample_intensity.plot.scatter(x=dates.name, y='median intensity',
                                              rot=90,
                                              fontsize='large',
                                              figsize=(20, 10),
                                              xticks=vaep.plotting.select_dates(
                                                  median_sample_intensity[dates.name])
                                              )
    fig = ax.get_figure()
    figures['median_scatter'] = folder_figures / 'median_scatter'
    vaep.savefig(fig, figures['median_scatter'])

# %% [markdown]
# - the closer the labels are there denser the samples are measured around that time.

# %% [markdown]
# ## Feature frequency  in data
#
# - higher count, higher probability to be sampled into training data
# - missing peptides are sampled both into training as well as into validation dataset
# - everything not in training data is validation data
#
# Based on unmodified training data

# %%
msg = "Total number of samples in training data split: {}"
print(msg.format(len(analysis.df)))

# %%
# # analysis.splits.to_wide_format()
# assert analysis.splits is splits, "Sanity check failed."

# %% [markdown]
# Recalculate feature frequency after selecting some samples

# %%
freq_per_feature = feature_frequency(analysis.df)
freq_per_feature

# %%
# freq_per_feature.name = 'Gene names freq' # name it differently?
freq_per_feature.to_json(folder_data / 'freq_features.json') # index.name is lost when data is stored
freq_per_feature.to_pickle(folder_data / 'freq_features.pkl')

# %% [markdown]
# Conserning sampling with frequency weights:
#   - larger weight -> higher probablility of being sampled
#   - weights need to be alignable to index of original DataFrame before grouping (same index)

# %% [markdown]
# ## Split: Train, validation and test data
#
# - test data is in clinical language often denoted as independent validation cohort
# - validation data (for model)

# %%
analysis.splits = DataSplits(is_wide_format=True)
splits = analysis.splits
print(f"{analysis.splits = }")
analysis.splits.__annotations__

# %% [markdown]
# ### Sample targets (Fake NAs)

# %% [markdown]
# Add goldstandard targets for valiation and test data
# - based on same day
# - same instrument

# %% [markdown]
# Create some target values by sampling 5% of the validation and test data.

# %%
analysis.to_long_format(inplace=True)
analysis.df_long

# %%
fake_na, splits.train_X = sample_data(analysis.df_long.squeeze(), sample_index_to_drop=0, weights=freq_per_feature, frac=0.1)
assert len(splits.train_X) > len(fake_na)
splits.val_y = fake_na.sample(frac=0.5).sort_index()
splits.test_y = fake_na.loc[fake_na.index.difference(splits.val_y.index)]
# splits

# %%
splits.test_y

# %%
splits.val_y

# %%
splits.train_X

# %% [markdown] tags=[]
# ### Save in long format
#
# - Data in long format: (peptide, sample_id, intensity)
# - no missing values kept

# %%
splits.dump(folder=folder_data, file_format=FILE_EXT)  # dumps data in long-format

# %%
# # Reload from disk
splits = DataSplits.from_folder(folder_data, file_format=FILE_EXT)

# %% [markdown]
# ## Save parameters

# %%
print(OmegaConf.to_yaml(params))

# %%
fname = folder_experiment/'data_config.yaml'
with open(fname, 'w') as f:
    OmegaConf.save(params, f)
fname

# %% [markdown]
# ## Saved Figures

# %%
# saved figures
figures
