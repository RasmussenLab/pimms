# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
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
# # Experiment 03 - Data
#
# Create data splits

# %%
from typing import Union
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

# %% [markdown]
# ## Arguments

# %% tags=["parameters"]
FN_INTENSITIES: str =  'data/single_datasets/df_intensities_proteinGroups_long_2017_2018_2019_2020_N05015_M04547/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# FN_PEPTIDE_FREQ: str = 'data/processed/count_all_peptides.json' # Peptide counts for all parsed files on erda (for data selection)
fn_rawfile_metadata: str = 'data/files_selected_metadata.csv' # Machine parsed metadata from rawfile workflow
# M: int = 5000 # M most common features
MIN_SAMPLE: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
min_RT_max: Union[int, float] = 120 # Minum retention time (RT) in minutes
index_col: Union[str,int] = ['Sample ID', 'Gene names'] # Can be either a string or position (typical 0 for first column)
# query expression for subsetting
# query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
# columns_name: str = 'Gene names'

# %%
# # peptides
# FN_INTENSITIES: str = 'data/single_datasets/df_intensities_peptides_long_2017_2018_2019_2020_N05011_M42725/Q_Exactive_HF_X_Orbitrap_Exactive_Series_slot_#6070.pkl'  # Intensities for feature
# index_col: Union[str,int] = ['Sample ID', 'peptide'] # Can be either a string or position (typical 0 for first column)
# folder_experiment: str = f'runs/experiment_03/{Path(FN_INTENSITIES).parent.name}/{Path(FN_INTENSITIES).stem}'
# %%
# # evidence


# %%
# There must be a better way...
@dataclass
class DataConfig:
    """Documentation. Copy pasted arguments to a dataclass."""
    FN_INTENSITIES: str  # Samples metadata extraced from erda
    file_ext: str # file extension
    # FN_PEPTIDE_FREQ: str # Peptide counts for all parsed files on erda (for data selection)
    fn_rawfile_metadata: str  # Machine parsed metadata from rawfile workflow
    # M: int # M most common features
    MIN_SAMPLE: Union[int, float] = 0.5 # Minimum number or fraction of total requested features per Sample
    index_col: Union[
        str, int
    ] = "Sample ID"  # Can be either a string or position (typical 0 for first column)
    # query expression for subsetting
    # query_subset_meta: str = "`instrument serial number` in ['Exactive Series slot #6070',]" # query for metadata, see files_selected_per_instrument_counts.csv for options
    logarithm: str = 'log2' # Log transformation of initial data (select one of the existing in numpy)
    folder_experiment: str = 'runs/experiment_03'
    # columns_name: str = "peptide"


params = DataConfig(
    FN_INTENSITIES=FN_INTENSITIES,
    file_ext=Path(FN_INTENSITIES).suffix[1:],
    # FN_PEPTIDE_FREQ=FN_PEPTIDE_FREQ,
    fn_rawfile_metadata=fn_rawfile_metadata,
    # M=M,
    MIN_SAMPLE=MIN_SAMPLE,
    index_col=index_col,
    # query_subset_meta=query_subset_meta,
    logarithm=logarithm,
    folder_experiment=folder_experiment,
    # columns_name=columns_name
)

params = OmegaConf.create(params.__dict__)
dict(params)

# %% [markdown]
# ## Setup

# %%
# if not folder_experiment:
#     folder_experiment = query_subset_meta.replace('_', ' ')
#     folder_experiment = parse_query_expression(Ffolder_experiment)
#     folder_experiment = folder_experiment.strip()
#     folder_experiment = folder_experiment.replace(' ', '_')
#     params.folder_experiment
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

# set automatically
# analysis.df.columns.name = params.columns_name
columns_name = analysis.df.columns.name # ToDo: MultiIndex adaptions will be needed

log_fct = getattr(np, params.logarithm)
analysis.log_transform(log_fct)
logger.info(f"{analysis = }")
analysis.df

# %% [markdown]
# ### Select M most common features
#
# - if number of features should be part of experiments, the selection has to be done here
# - select between `analysis.M` (number of features available) and requested number of features `params.M` 
# - can be random or based on most common features from counter objects
#
# > Ignored for now, instead select based on feature availabiltiy across samples (see below)

# %% tags=[]
# import json
# from collections import Counter
# # Use PeptideCounter instead?
# with open(Path(params.FN_PEPTIDE_FREQ)) as f:
#     freq_pep_all = Counter(json.load(f)['counter'])
    
# selected_peptides = {k: v for k, v in freq_pep_all.most_common(params.M)}
# print(f"No. of selected features: {len(selected_peptides):,d}")
analysis.M

# %% [markdown]
# ### Sample selection
#
# - ensure unique indices

# %%
assert analysis.df.index.is_unique, "Duplicates in index"
analysis.df.sort_index(inplace=True)

# %% [markdown]
# Select samples based on completeness

# %%
if isinstance(params.MIN_SAMPLE, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {params.MIN_SAMPLE}\n'
    # assumes df in wide format
    params.MIN_SAMPLE = int(analysis.df.shape[1] * params.MIN_SAMPLE)
    msg += f'This translates to a minimum number of total samples: {params.MIN_SAMPLE}'
    print(msg)

sample_counts = analysis.df.notna().sum(axis=1) # if DataFrame

mask = sample_counts > params.MIN_SAMPLE
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
print(msg)
analysis.df = analysis.df.loc[mask]

# %%
params.used_samples = analysis.df.index.to_list()

# %% [markdown]
# ## Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %%
df_meta = pd.read_csv('data/files_selected_metadata.csv', index_col=0)
df_meta = df_meta.loc[analysis.df.index.to_list()] # index is sample index
date_col = 'Content Creation Date'
df_meta[date_col] = pd.to_datetime(df_meta[date_col])
df_meta

# %%
cols_instrument = thermo_raw_files.cols_instrument
df_meta.groupby(cols_instrument)[date_col].agg(['count','min','max']) 

# %%
df_meta.describe(datetime_is_numeric=True, percentiles=np.linspace(0.05, 0.95, 10))

# %% [markdown]
# set a minimum retention time

# %%
# min_RT_max = 120 # minutes
msg = f"Minimum RT time maxiumum is set to {min_RT_max} minutes (to exclude too short runs, which are potentially fractions)."
mask_RT = df_meta['MS max RT'] >= 120 # can be integrated into query string
msg += f" Total number of samples retained: {int(mask_RT.sum())}."
print(msg)

# %%
df_meta = df_meta.loc[mask_RT]
df_meta = df_meta.sort_values(date_col)

# %%
meta_stats = df_meta.describe(include='all', datetime_is_numeric=True)
meta_stats

# %% [markdown]
# subset with variation

# %%
meta_stats.loc[:, (meta_stats.loc['unique'] > 1) |  (meta_stats.loc['std'] > 0.1)]

# %% [markdown]
# check some columns describing settings
#   - quite some variation due to `MS max charge`: Is it a parameter?

# %%
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

# %% [markdown]
# view without `MS max charge`:
#   - software can be updated
#   - variation by `injection volume setting` and instrument over time
#     - 500ng of peptides should be injected, based on concentration of peptides this setting is adjustd to get it
#   - missing `dilution factor`
#   

# %%
to_drop = ['MS max charge']
df_meta[meta_raw_settings].drop(to_drop, axis=1).drop_duplicates() # index gives first example with this combination


# %% [markdown]
# - check for variation in `software Version` and `injection volume setting`
#
#
# Update selection of samples based on metadata (e.g. minimal retention time)

# %%
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

# %% [markdown]
# ### Interactive and Single plots

# %% [markdown]
# Scatter plots need to become interactive.

# %%
sample_counts.name = 'identified features'

# %%
K = 2
pcs = analysis.get_PCA(n_components=K) # should be renamed to get_PCs
pcs = pcs.iloc[:,:K].join(df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs = pcs.reset_index()
pcs

# %%
pcs.describe(include='all')

# %%
fig, ax = plt.subplots(figsize=(18,10))
analyzers.seaborn_scatter(pcs[pcs_name], fig, ax, meta=pcs['Thermo Scientific instrument model'])

# %%
fig, ax = plt.subplots(figsize=(23,10))
analyzers.plot_date_map(pcs[pcs_name], fig, ax, pcs[date_col])
vaep.savefig(fig, folder_figures / 'pca_sample_by_date')

# %% [markdown]
# - software version: Does it make a difference?
# - size: number of features in a single sample

# %%
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

# %% [markdown]
# ## Sample Medians and percentiles
#
# - see boxplot [function in R](https://github.dev/symbioticMe/proBatch/blob/8fae15049be67693bd0d4a4383b51bfb4fb287a6/R/initial_assessment.R#L199-L317), which is used for [publication figure](https://github.dev/symbioticMe/batch_effects_workflow_code/blob/696eb609a55ba9ece68b732e616d7ebeaa660373/AgingMice_study/analysis_AgingMice/5b_Fig2_initial_assessment_normalization.R)
# - check boxplot functions: [bokeh](https://docs.bokeh.org/en/latest/docs/gallery/boxplot.html), [plotly](https://plotly.com/python/box-plots/), [eventplot](https://matplotlib.org/stable/gallery/lines_bars_and_markers/eventplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-eventplot-demo-py)

# %%
# analysis.df.iloc[:100].T.boxplot(rot=90, backend=None, figsize=None)
# analysis.df.boxplot()
analysis.df.head()

# %%
df = analysis.df
df = df.join(df_meta[date_col])
df = df.set_index(date_col).sort_index().to_period('min').T
df

# %%
ax = df.boxplot(rot=80, figsize=(20, 10), fontsize='large', showfliers=False, showcaps=False)
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
vaep.savefig(fig, folder_figures / 'median_boxplot')
figures['median_boxplot'] = fig


# %% [markdown]
# Plot sample median over time
#   - check if points are equally spaced (probably QC samples are run in close proximity)
#   - the machine will be not use for intermediate periods

# %%
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


# %% [markdown]
# - the closer the labels are there denser the samples are measured aroudn that time.

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

# %%
# percentiles = (0.8, 0.9)  # change here

# percent_str = [f'{int(x*100)}%' for x in percentiles]
# split_at_date = analysis.df_meta[date_col].describe(
#     datetime_is_numeric=True, percentiles=(0.8, 0.9)).loc[percent_str]
# split_at_date = tuple(pd.Timestamp(t.date()) for t in split_at_date)

# print(f"{split_at_date[0] = }", f"{split_at_date[1] = }", sep="\n")

# %%
# idx_train = analysis.df_meta[date_col] < split_at_date[0]
# analysis.splits.train_X = analysis.df.loc[idx_train]
# analysis.splits.train_X

# %%
# idx_validation = ((analysis.df_meta[date_col] >= split_at_date[0]) & (
#     analysis.df_meta[date_col] < split_at_date[1]))
# analysis.splits.val_X = analysis.df.loc[idx_validation]
# analysis.splits.val_X

# %%
# idx_test = (analysis.df_meta[date_col] >= split_at_date[1])
# # analysis.df_test =
# analysis.splits.test_X = analysis.df.loc[idx_test]
# analysis.splits.test_X

# %%
# idx_test_na = analysis.splits.test_X.stack(
#     dropna=False).loc[splits.test_X.isna().stack()].index
# print(f"number of missing values in test data: {len(idx_test_na)}")

# %% [markdown]
# ## Peptide frequency  in training data
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
freq_per_feature.to_json(folder_data / 'freq_train.json')

# %% [markdown]
# Conserning sampling with frequency weights:
#   - larger weight -> higher probablility of being sampled
#   - weights need to be alignable to index of original DataFrame before grouping (same index)

# %% [markdown]
# ## Sample targets (Fake NAs)

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
# analysis.splits.to_long_format(name_values='intensity') # long format as sample_data uses long-format 
# analysis.splits
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
# potential bug: wrong order...
# splits.val_X, splits.val_y = sample_data(splits.val_X, sample_index_to_drop=0, weights=freq_per_peptide) # I this the wrong way around?
# splits.test_X, splits.test_y = sample_data(splits.test_X, sample_index_to_drop=0, weights=freq_per_peptide)

# for k, s in splits:
#     s.sort_index(inplace=True)

# %% [markdown] tags=[]
# ## Save in long format
#
# - Data in long format: (peptide, sample_id, intensity)
# - no missing values kept

# %%
splits.dump(folder=folder_data, file_format=FILE_EXT)  # dumps data in long-format

# %%
# # Reload from disk
splits = DataSplits.from_folder(folder_data, file_format=FILE_EXT)

# %% [markdown]
# ## PCA plot of training data
#
# - [ ] update: uses old metadata reading to indicate metadata derived from filename
#

# %%
ana_train_X = analyzers.AnalyzePeptides(data=splits.train_X, is_wide_format=False, ind_unstack=columns_name)
figures['pca_train'] = ana_train_X.plot_pca()
vaep.savefig(figures['pca_train'], folder_figures / f'pca_plot_raw_data_{ana_train_X.fname_stub}')
# ana_train_X = add_meta_data(ana_train_X) # inplace, returns ana_train_X

# %% [markdown]
# ## Move: Script on matching similar samples 
#
# - how to identify similar samples, e.g. using KNNs

# %%
# # add to DataSplits a inputs attribute

# data_dict = {'train': splits.train_X, 'valid': splits.val_X, 'test': splits.test_X}
# PCs = pd.DataFrame()
# split_map = pd.Series(dtype='string')
# for key, df in data_dict.items():
#     df = df.unstack()
#     PCs = PCs.append(ana_train_X.calculate_PCs(df))
#     split_map = split_map.append(pd.Series(key, index=df.index))

# fig, ax = plt.subplots(figsize=(15,8))
# ax.legend(title='splits')
# analyzers.seaborn_scatter(PCs.iloc[:, :2], fig, ax, meta=split_map,
#                           title='First two principal compements (based on training data PCA)')
# ax.get_legend().set_title("split")

# %% [markdown]
# For *Collaborative Filtering*, new samples could be initialized based on a KNN approach in the original sample space or the reduced PCA dimension.
#   - The sample embeddings of the K neighearst neighbours could be averaged for a new sample

# %%
# # Optional: Change number of principal components
# # K = 2
# # _ = ana_train_X.get_PCA(n_components=K)

# train_PCs = ana_train_X.calculate_PCs(splits.train_X.unstack())
# test_PCs = ana_train_X.calculate_PCs(splits.test_X.unstack())
# nn = NearestNeighbors(n_neighbors=5).fit(train_PCs)

# %% [markdown]
# Select K neareast neighbors for first test data sample from training data. Compare equal distance mean to mean weighted by distances.

# %%
# d, idx = nn.kneighbors(test_PCs.iloc[1:2])
# # test_PCs.iloc[1]
# idx

# %%
# train_PCs.iloc[idx[0]]

# %%
# w = d / d.sum()
# display(f"Sample weights based on distances: {w = }")
# w.flatten().reshape(5,1) * train_PCs.iloc[idx[0]] # apply weights to values

# %%
# pd.DataFrame( (train_PCs.iloc[idx[0]].mean(), # mean
#               (w.flatten().reshape(5,1) * train_PCs.iloc[idx[0]]).sum() # sum of weighted samples
#               ), index=['mean','weighted by distance '])

# %% [markdown]
# Add visual representation of picked points in the first two principal components

# %%
# ax.scatter(x=test_PCs.iloc[1]['PC 1'], y=test_PCs.iloc[1]['PC 2'], s=100, marker="v", c='r')
# ax.scatter(x=train_PCs.iloc[idx[0]]['PC 1'], y=train_PCs.iloc[idx[0]]['PC 2'], s=100, marker="s", c='y')
# fig

# %% [markdown]
# ## Save parameters

# %%
print(OmegaConf.to_yaml(params))

# %%
with open(folder_experiment/'data_config.yaml', 'w') as f:
    OmegaConf.save(params, f)
