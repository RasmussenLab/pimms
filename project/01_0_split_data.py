# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
from pathlib import Path

from typing import Union, List


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

import vaep
from vaep.io.datasplits import DataSplits
from vaep.io import thermo_raw_files
from vaep.sampling import feature_frequency, sample_data

from vaep.analyzers import analyzers
from vaep.analyzers.analyzers import  AnalyzePeptides

logger = vaep.logging.setup_nb_logger()
logger.info("Split data and make diagnostic plots")

def add_meta_data(analysis: AnalyzePeptides, df_meta: pd.DataFrame):
    try:
        analysis.df = analysis.df.loc[df_meta.index]
    except KeyError as e:
        logger.warning(e)
        logger.warning("Ignore missing samples in quantified samples")
        analysis.df = analysis.df.loc[analysis.df.index.intersection(
            df_meta.index)]

    analysis.df_meta = df_meta
    return analysis


pd.options.display.max_columns = 32
plt.rcParams['figure.figsize'] = [4, 2]
vaep.plotting.make_large_descriptors(5)

figures = {}  # collection of ax or figures
dumps = {}  # collection of data dumps

# %% [markdown]
# ## Arguments

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
# Sample (rows) intensiites for features (columns)
FN_INTENSITIES: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
# Can be either a string or position (typical 0 for first column), or a list of these.
index_col: Union[str, int] = 0
# wide_format: bool = False # intensities in wide format (more memory efficient of csv). Default is long_format (more precise)
# Manuelly set column names (of Index object in columns)
column_names: List[str] = ["Gene Names"]
# Machine parsed metadata from raw file (see workflows/metadata), wide format per sample
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'
# Minimum number or fraction of feature prevalence across samples to be kept
feat_prevalence: Union[int, float] = 0.25
# Minimum number or fraction of total requested features per Sample
sample_completeness: Union[int, float] = 0.5
select_N: int = None  # only use latest N samples
sample_N: bool = False # if select_N, sample N randomly instead of using latest?
random_state: int = 42  # random state for reproducibility of splits
# based on raw file meta data, only take samples with RT > min_RT_time
min_RT_time: Union[int, float] = None
# Log transformation of initial data (select one of the existing in numpy)
logarithm: str = 'log2'
folder_experiment: str = f'runs/example'
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
# metadata -> defaults for metadata extracted from machine data, used for plotting
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data


# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
params = vaep.nb.args_from_dict(args)
params


# %%
if isinstance(params.index_col, str) or isinstance(params.index_col, int):
    params.overwrite_entry('index_col', [params.index_col])
params.index_col  # make sure it is an iterable

# %% [markdown]
# ## Raw data

# %% [markdown]
# process arguments

# %%
logger.info(f"{params.FN_INTENSITIES = }")


FILE_FORMAT_TO_CONSTRUCTOR = {'csv': 'from_csv',
                              'pkl': 'from_pickle',
                              'pickle': 'from_pickle',
                              }

FILE_EXT = Path(params.FN_INTENSITIES).suffix[1:]
logger.info(
    f"File format (extension): {FILE_EXT}  (!specifies data loading function!)")

# %%
# AnalyzePeptides.from_csv
constructor = getattr(AnalyzePeptides, FILE_FORMAT_TO_CONSTRUCTOR[FILE_EXT])
analysis = constructor(fname=params.FN_INTENSITIES,
                       index_col=params.index_col,
                       )
if params.column_names:
    analysis.df.columns.names = params.column_names

if not analysis.df.index.name:
    logger.warning("No sample index name found, setting to 'Sample ID'")
    analysis.df.index.name = 'Sample ID'

log_fct = getattr(np, params.logarithm)
analysis.log_transform(log_fct)
logger.info(f"{analysis = }")
analysis.df

# %%
ax = analysis.df.notna().sum(axis=0).to_frame(
    analysis.df.columns.name).plot.box()
ax.set_ylabel('number of observation across samples')


# %%
fname = params.out_folder / '01_0_data_stats.xlsx'
dumps[fname.name] = fname.as_posix()
writer = pd.ExcelWriter(fname)

notna = analysis.df.notna()
data_stats_original = pd.concat(
    [
        notna.sum().describe().rename('feat_stats'),
        notna.sum(axis=1).describe().rename('sample_stats')
    ],
    axis=1)
data_stats_original.to_excel(writer, sheet_name='data_stats_original')
data_stats_original


# %% [markdown]
# In case there are multiple features for each intensity values (currenlty: peptide sequence and charge), combine the column names to a single str index.
#
# > The Collaborative Modeling approach will need a single feature column.

# %%
def join_as_str(seq):
    ret = "_".join(str(x) for x in seq)
    return ret

# ToDo: join multiindex samples indices (pkl dumps)
# if hasattr(analysis.df.columns, "levels"):
if isinstance(analysis.df.columns, pd.MultiIndex):
    logger.warning("combine MultiIndex columns to one feature column")
    print(analysis.df.columns[:10].map(join_as_str))
    _new_name = join_as_str(analysis.df.columns.names)
    analysis.df.columns = analysis.df.columns.map(join_as_str)
    analysis.df.columns.name = _new_name
    logger.warning(f"New name: {analysis.df.columns.names = }")

# %% [markdown]
# ## Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %%
if params.fn_rawfile_metadata:
    df_meta = pd.read_csv(params.fn_rawfile_metadata, index_col=0)
else:
    logger.warning(f"No metadata for samples provided, create placeholder.")
    if params.meta_date_col:
        raise ValueError(
            f"No metadata provided, but data column set: {params.meta_date_col}")
    if params.meta_cat_col:
        raise ValueError(
            f"No metadata provided, but data column set: {params.meta_cat_col}")
    df_meta = pd.DataFrame(index=analysis.df.index)
df_meta = df_meta.loc[analysis.df.index.to_list()]  # index is sample index
if df_meta.index.name is None:
    df_meta.index.name = params.index_col[0]
df_meta

# %%
if params.meta_date_col:
    df_meta[params.meta_date_col] = pd.to_datetime(
        df_meta[params.meta_date_col])
else:
    params.overwrite_entry('meta_date_col', 'PlaceholderTime')
    df_meta[params.meta_date_col] = range(len(df_meta))
df_meta

# %%
if df_meta.columns.isin(thermo_raw_files.cols_instrument).sum() == len(thermo_raw_files.cols_instrument):
    display(df_meta.groupby(thermo_raw_files.cols_instrument)[
            params.meta_date_col].agg(['count', 'min', 'max']))
else:
    logger.info(
        f"Instrument column not found: {thermo_raw_files.cols_instrument}")

# %%
df_meta.describe(datetime_is_numeric=True,
                 percentiles=np.linspace(0.05, 0.95, 10))

# %% [markdown]
# select samples with a minimum retention time

# %%
if params.min_RT_time:
    logger.info(
        "Metadata should have 'MS max RT' entry from ThermoRawFileParser")
    msg = f"Minimum RT time maxiumum is set to {params.min_RT_time} minutes (to exclude too short runs, which are potentially fractions)."
    # can be integrated into query string
    mask_RT = df_meta['MS max RT'] >= params.min_RT_time
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
    display(meta_stats.loc[:, (meta_stats.loc['unique']
            > 1) | (meta_stats.loc['std'] > 0.1)])
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
        # index gives first example with this combination
        df_meta[meta_raw_settings].drop_duplicates()
    )


# %% [markdown]
# - check for variation in `software Version` and `injection volume setting`
#
#
# Update selection of samples based on metadata (e.g. minimal retention time)
# - sort data the same as sorted meta data

# %%
analysis = add_meta_data(analysis, df_meta=df_meta)

# %% [markdown]
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
    if params.sample_N:
        analysis.df_meta = analysis.df_meta.sample(params.select_N)
    else:
        analysis.df_meta = analysis.df_meta.iloc[-params.select_N:]

    analysis.df = analysis.df.loc[analysis.df_meta.index].dropna(
        how='all', axis=1)
    ax = analysis.df.T.describe().loc['count'].hist()
    _ = ax.set_title('histogram of features for all eligable samples')

# %% [markdown]
# ## First Step: Select features by prevalence
# - `feat_prevalence` across samples

# %%
freq_per_feature = analysis.df.notna().sum()  # on wide format
if isinstance(params.feat_prevalence, float):
    N_samples = len(analysis.df_meta)
    logger.info(f"Current number of samples: {N_samples}")
    logger.info(
        f"Feature has to be present in at least {params.feat_prevalence:.2%} of samples")
    params.overwrite_entry('feat_prevalence', int(
        N_samples * params.feat_prevalence))
assert isinstance(params.feat_prevalence, int)
# ! check that feature prevalence is greater equal to 3 (otherwise train, val, test split is not possible)
logger.info(
    f"Feature has to be present in at least {params.feat_prevalence} of samples")
# select features
mask = freq_per_feature >= params.feat_prevalence
logger.info(f"Drop {(~mask).sum()} features")
freq_per_feature = freq_per_feature.loc[mask]
analysis.df = analysis.df.loc[:, mask]
analysis.N, analysis.M = analysis.df.shape

# # potentially create freq based on DataFrame
analysis.df

notna = analysis.df.notna()
data_stats_filtered = pd.concat(
    [
        notna.sum().describe().rename('feat_stats'),
        notna.sum(axis=1).describe().rename('sample_stats')
    ],
    axis=1)
data_stats_filtered.to_excel(writer, sheet_name='data_stats_filtered')
data_stats_filtered

# %% [markdown]
# ## Second step - Sample selection

# %% [markdown]
# Select samples based on completeness

# %%
if isinstance(params.sample_completeness, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {params.sample_completeness}\n'
    # assumes df in wide format
    params.overwrite_entry('sample_completeness', int(
        analysis.df.shape[1] * params.sample_completeness))
    msg += f'This translates to a minimum number of features per sample (to be included): {params.sample_completeness}'
    logger.info(msg)

sample_counts = analysis.df.notna().sum(axis=1)  # if DataFrame
sample_counts.describe()

# %%
mask = sample_counts > params.sample_completeness
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
print(msg)
analysis.df = analysis.df.loc[mask]
analysis.df = analysis.df.dropna(
    axis=1, how='all')  # drop now missing features

# %%
params.N, params.M = analysis.df.shape  # save data dimensions
params.used_samples = analysis.df.index.to_list()

# %% [markdown]
# ### Histogram of features per sample

# %%
ax = analysis.df.notna().sum(axis=1).hist()
ax.set_xlabel('features per eligable sample')
ax.set_ylabel('observations')
fname = params.out_figures / 'hist_features_per_sample'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = analysis.df.notna().sum(axis=0).sort_values().plot()
_new_labels = [l.get_text().split(';')[0] for l in ax.get_xticklabels()]
_ = ax.set_xticklabels(_new_labels, rotation=45,
                       horizontalalignment='right')
ax.set_xlabel('feature prevalence')
ax.set_ylabel('observations')
fname = params.out_figures / 'feature_prevalence'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)


# %% [markdown]
# ### Number off observations accross feature value

# %%
min_max = vaep.plotting.data.min_max(analysis.df.stack())
ax, bins = vaep.plotting.data.plot_histogram_intensities(
    analysis.df.stack(), min_max=min_max)

fname = params.out_figures / 'intensity_distribution_overall'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=analysis.df, type='scatter')
fname = params.out_figures / 'intensity_median_vs_prop_missing_scatter'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=analysis.df, type='boxplot')
fname = params.out_figures / 'intensity_median_vs_prop_missing_boxplot'
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
pcs = analysis.get_PCA(n_components=K)  # should be renamed to get_PCs
pcs = pcs.iloc[:, :K].join(analysis.df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs_index_name = pcs.index.name
pcs = pcs.reset_index()
pcs

# %%
pcs.describe(include='all', datetime_is_numeric=True).T

# %%
if params.meta_cat_col:
    fig, ax = plt.subplots(figsize=(2,2))
    analyzers.seaborn_scatter(
        pcs[pcs_name], ax, meta=pcs[params.meta_cat_col], title=f"by {params.meta_cat_col}")
    fname = (params.out_figures
              / f'pca_sample_by_{"_".join(params.meta_cat_col.split())}')
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %%
if params.meta_date_col != 'PlaceholderTime':
    fig, ax = plt.subplots()
    analyzers.plot_date_map(
        df=pcs[pcs_name], ax=ax, dates=pcs[params.meta_date_col], title=f'by {params.meta_date_col}')
    fname = params.out_figures / 'pca_sample_by_date'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
# - software version: Does it make a difference?
# - size: number of features in a single sample

# %%
fig, ax = plt.subplots()
col_identified_feat = 'identified features'
analyzers.plot_scatter(
    pcs[pcs_name],
    ax,
    pcs[col_identified_feat],
    title=f'by {col_identified_feat}',
    size=5,
)
fname = (params.out_figures
         / f'pca_sample_by_{"_".join(col_identified_feat.split())}.pdf')
figures[fname.stem] = fname
vaep.savefig(fig, fname)

# %%
fig = px.scatter(
    pcs, x=pcs_name[0], y=pcs_name[1],
    hover_name=pcs_index_name,
    # hover_data=analysis.df_meta,
    title=f'First two Principal Components of {analysis.M} features for {pcs.shape[0]} samples',
    # color=pcs['Software Version'],
    color=col_identified_feat,
    template='none',
    width=1200, # 4 inches x 300 dpi
    height=600 # 2 inches x 300 dpi
)
fname = (params.out_figures
         / f'pca_sample_by_{"_".join(col_identified_feat.split())}_plotly.pdf')
figures[fname.stem] = fname
fig.write_image(fname)
fig  # stays interactive in html

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
ax = df.boxplot(rot=80, figsize=(8, 3), fontsize=5,
                showfliers=False, showcaps=False)
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
fname = params.out_figures / 'median_boxplot'
figures[fname.stem] = fname
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
                                              figsize=(8, 2),
                                              s=5,
                                              xticks=vaep.plotting.select_dates(
                                                  median_sample_intensity[dates.name])
                                              )
    fig = ax.get_figure()
    figures['median_scatter'] = params.out_figures / 'median_scatter'
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
# index.name is lost when data is stored
fname = params.data / 'freq_features.json'
dumps[fname.name] = fname
freq_per_feature.to_json(fname)
fname = fname.with_suffix('.pkl')
dumps[fname.name] = fname
freq_per_feature.to_pickle(fname)

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
analysis.splits = DataSplits(is_wide_format=False)
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
fake_na, splits.train_X = sample_data(analysis.df_long.squeeze(),
                                      sample_index_to_drop=0,
                                      weights=freq_per_feature,
                                      frac=0.1,
                                      random_state=params.random_state,)
assert len(splits.train_X) > len(fake_na)
splits.val_y = fake_na.sample(frac=0.5, random_state=params.random_state).sort_index()
splits.test_y = fake_na.loc[fake_na.index.difference(splits.val_y.index)]
# splits

# %%
splits.test_y

# %%
splits.val_y

# %%
splits.train_X

# %%
# ToDo check that feature indices and sample indicies overlap
# -> a single feature cannot be only in the validation or test split
# -> single features should be put into the training data
# -> or raise error as feature completness treshold is so low that less than 3 samples
# per feature are allowd.

diff = (splits
    .val_y
    .index
    .levels[-1]
    .difference(splits
                .train_X
                .index
                .levels[-1]
    ).to_list())
if diff:
    to_remove = splits.val_y.loc[pd.IndexSlice[:, diff]]
    display(to_remove)
    splits.train_X = pd.concat([splits.train_X, to_remove])
    splits.val_y = splits.val_y.drop(to_remove.index)
diff

# %%
diff = (splits
    .test_y
    .index
    .levels[-1]
    .difference(splits
                .train_X
                .index
                .levels[-1]
    ).to_list())
if diff:
    to_remove = splits.test_y.loc[pd.IndexSlice[:, diff]]
    display(to_remove)
    splits.train_X = pd.concat([splits.train_X, to_remove])
    splits.test_y = splits.test_y.drop(to_remove.index)
diff


# %% [markdown]
# ### Save in long format
#
# - Data in long format: (peptide, sample_id, intensity)
# - no missing values kept

# %%
# dumps data in long-format
splits_dumped = splits.dump(folder=params.data, file_format=params.file_format)
dumps.update(splits_dumped)
splits_dumped

# %% [markdown]
# ### Reload from disk

# %%
splits = DataSplits.from_folder(params.data, file_format=params.file_format)

# %% [markdown]
# ## plot distribution of splits

# %%
splits_df = pd.DataFrame(index=analysis.df_long.index)
splits_df['train'] = splits.train_X
splits_df['val'] = splits.val_y
splits_df['test'] = splits.test_y
stats_splits = splits_df.describe()
stats_splits.to_excel(writer, 'stats_splits', float_format='%.2f')
stats_splits

# %%
# whitespaces in legends are not displayed correctly...
# max_int_len   = len(str(int(stats_splits.loc['count'].max()))) +1
# _legend = [
#     f'{s:<5} (N={int(stats_splits.loc["count", s]):>{max_int_len},d})'.replace(
#         ' ', '\u00A0')
#     for s in ('train', 'val', 'test')]
_legend = [
    f'{s:<5} (N={int(stats_splits.loc["count", s]):,d})'
    for s in ('train', 'val', 'test')]
print(_legend)

# %%
ax = (splits
      .train_X
      .plot
      .hist(
          bins=bins,
          ax=None,
          color='C0',
))
_ = (splits
     .val_y
     .plot
     .hist(bins=bins,
           xticks=list(bins),
           ax=ax,
           color='C2',
           legend=True)
     )
ax.legend(_legend[:-1])
fname = params.out_figures / 'test_over_train_split.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
min_bin, max_bin = vaep.plotting.data.min_max(splits.val_y)
bins = range(int(min_bin), int(max_bin), 1)
ax = splits_df.plot.hist(bins=bins,
                         xticks=list(bins),
                         legend=False,
                         stacked=True,
                         color=['C0', 'C1', 'C2'],
    )
ax.legend(_legend)
ax.set_xlabel('Intensity bins')
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = params.out_figures / 'splits_freq_stacked.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = splits_df.drop('train', axis=1).plot.hist(bins=bins,
                                               xticks=list(bins),
                                               color=['C1', 'C2'],
                                               legend=False,
                                               stacked=True,
                        )
ax.legend(_legend[1:])
ax.set_xlabel('Intensity bins')
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = params.out_figures / 'val_test_split_freq_stacked_.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# plot training data missing plots

# %%
splits.to_wide_format()

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='scatter')
fname = params.out_figures / 'intensity_median_vs_prop_missing_scatter_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='boxplot')
fname = params.out_figures / 'intensity_median_vs_prop_missing_boxplot_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ## Save parameters

# %%
fname = params.folder_experiment / 'data_config.yaml'
params.dump(fname)
params

# %% [markdown]
# ## Saved Figures

# %%
# saved figures
figures

# %% [markdown]
# Saved dumps

# %%
writer.close()
dumps
# %%
