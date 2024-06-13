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
# # Experiment 03 - Data
#
# Create data splits

# %% tags=["hide-input"]
import logging
from functools import partial
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display
from sklearn.model_selection import train_test_split

import vaep
import vaep.io.load
from vaep.analyzers import analyzers
from vaep.io.datasplits import DataSplits
from vaep.sampling import feature_frequency
from vaep.sklearn import get_PCA

logger = vaep.logging.setup_nb_logger()
logger.info("Split data and make diagnostic plots")
logging.getLogger('fontTools').setLevel(logging.WARNING)


def align_meta_data(df: pd.DataFrame, df_meta: pd.DataFrame):
    try:
        df = df.loc[df_meta.index]
    except KeyError as e:
        logger.warning(e)
        logger.warning("Ignore missing samples in quantified samples")
        df = df.loc[df.index.intersection(
            df_meta.index)]
    return df_meta


pd.options.display.max_columns = 32
plt.rcParams['figure.figsize'] = [4, 2]

vaep.plotting.make_large_descriptors(7)

figures = {}  # collection of ax or figures
dumps = {}  # collection of data dumps

# %% [markdown]
# ## Arguments

# %% tags=["hide-input"]
# catch passed parameters
args = None
args = dict(globals()).keys()

# %% tags=["parameters"]
FN_INTENSITIES: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'  # Sample (rows), features (columns)
index_col: Union[str, int] = 0  # Can be either a string or position (default 0 for first column), or a list of these.
column_names: List[str] = ["Gene Names"]  # Manuelly set column names (of Index object in columns)
fn_rawfile_metadata: str = 'data/dev_datasets/HeLa_6070/files_selected_metadata_N50.csv'  # metadata for samples (rows)
feat_prevalence: Union[int, float] = 0.25  # Minimum number or fraction of feature prevalence across samples to be kept
sample_completeness: Union[int, float] = 0.5  # Minimum number or fraction of total requested features per Sample
select_N: int = None  # only use latest N samples
sample_N: bool = False  # if select_N, sample N randomly instead of using latest N
random_state: int = 42  # random state for reproducibility of splits
logarithm: str = 'log2'  # Log transformation of initial data (select one of the existing in numpy)
folder_experiment: str = 'runs/example'  # folder to save figures and data dumps
folder_data: str = ''  # specify special data directory if needed
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
use_every_nth_xtick: int = 1  # use every nth xtick in plots (default 1, i.e. every xtick is kept)
# metadata -> defaults for metadata extracted from machine data, used for plotting
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data
# train, validation and test data splits
frac_non_train: float = 0.1  # fraction of non training data (validation and test split)
frac_mnar: float = 0.0  # fraction of missing not at random data, rest: missing completely at random
prop_sample_w_sim: float = 1.0  # proportion of samples with simulated missing values
feat_name_display: str = None  # display name for feature name (e.g. 'protein group')


# %% tags=["hide-input"]
args = vaep.nb.get_params(args, globals=globals())
args

# %% tags=["hide-input"]
args = vaep.nb.args_from_dict(args)
args

# %% tags=["hide-input"]
if not 0.0 <= args.frac_mnar <= 1.0:
    raise ValueError("Invalid MNAR float value (should be betw. 0 and 1):"
                     f" {args.frac_mnar}")

if isinstance(args.index_col, str) or isinstance(args.index_col, int):
    args.overwrite_entry('index_col', [args.index_col])
args.index_col  # make sure it is an iterable

# %% [markdown]
# ## Raw data

# %% [markdown]
# process arguments

# %% tags=["hide-input"]
logger.info(f"{args.FN_INTENSITIES = }")


FILE_FORMAT_TO_CONSTRUCTOR = {'csv': 'from_csv',
                              'pkl': 'from_pickle',
                              'pickle': 'from_pickle',
                              }

FILE_EXT = Path(args.FN_INTENSITIES).suffix[1:]
logger.info(
    f"File format (extension): {FILE_EXT}  (!specifies data loading function!)")

# %% tags=["hide-input"]
# # ! factor out file reading to a separate module, not class
# AnalyzePeptides.from_csv
constructor = getattr(vaep.io.load, FILE_FORMAT_TO_CONSTRUCTOR[FILE_EXT])
df = constructor(fname=args.FN_INTENSITIES,
                 index_col=args.index_col,
                 )
if args.column_names:
    df.columns.names = args.column_names
if args.feat_name_display is None:
    args.overwrite_entry('feat_name_display', 'features')
    if args.column_names:
        args.overwrite_entry('feat_name_display', args.column_names[0])


if not df.index.name:
    logger.warning("No sample index name found, setting to 'Sample ID'")
    df.index.name = 'Sample ID'

if args.logarithm:
    log_fct = getattr(np, args.logarithm)
    df = log_fct(df)  # ! potentially add check to increase value by 1 if 0 is present (should be part of preprocessing)
df

# %% tags=["hide-input"]
ax = (df
      .notna()
      .sum(axis=0)
      .to_frame(df.columns.name)
      .plot
      .box()
      )
ax.set_ylabel('Frequency')


# %% tags=["hide-input"]
fname = args.out_folder / '01_0_data_stats.xlsx'
dumps[fname.name] = fname.as_posix()
writer = pd.ExcelWriter(fname)

notna = df.notna()
data_stats_original = pd.concat(
    [
        notna.sum().describe().rename('feat_stats'),
        notna.sum(axis=1).describe().rename('sample_stats')
    ],
    axis=1)
data_stats_original.to_excel(writer, sheet_name='data_stats_original')
data_stats_original


# %% [markdown]
# In case there are multiple features for each intensity values (currenlty: peptide sequence and charge),
# combine the column names to a single str index.
#
# > The Collaborative Modeling approach will need a single feature column.

# %% tags=["hide-input"]
def join_as_str(seq):
    ret = "_".join(str(x) for x in seq)
    return ret


if isinstance(df.columns, pd.MultiIndex):
    logger.warning("combine MultiIndex columns to one feature column")
    print(df.columns[:10].map(join_as_str))
    _new_name = join_as_str(df.columns.names)
    df.columns = df.columns.map(join_as_str)
    df.columns.name = _new_name
    logger.warning(f"New name: {df.columns.names = }")

# %% [markdown]
# ## Machine metadata
#
# - read from file using [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)

# %% tags=["hide-input"]
if args.fn_rawfile_metadata:
    df_meta = pd.read_csv(args.fn_rawfile_metadata, index_col=0)
else:
    logger.warning("No metadata for samples provided, create placeholder.")
    if args.meta_date_col:
        raise ValueError(
            f"No metadata provided, but data column set: {args.meta_date_col}")
    if args.meta_cat_col:
        raise ValueError(
            f"No metadata provided, but data column set: {args.meta_cat_col}")
    df_meta = pd.DataFrame(index=df.index)
df_meta = df_meta.loc[df.index.to_list()]  # index is sample index
if df_meta.index.name is None:
    df_meta.index.name = args.index_col[0]
df_meta

# %% tags=["hide-input"]
if args.meta_date_col:
    df_meta[args.meta_date_col] = pd.to_datetime(
        df_meta[args.meta_date_col])
else:
    args.overwrite_entry('meta_date_col', 'PlaceholderTime')
    df_meta[args.meta_date_col] = range(len(df_meta))
df_meta


# %% tags=["hide-input"]
df_meta.describe(percentiles=np.linspace(0.05, 0.95, 10))

# %% tags=["hide-input"]
df_meta = df_meta.sort_values(args.meta_date_col)

# %% tags=["hide-input"]
meta_stats = df_meta.describe(include='all')
meta_stats

# %% [markdown]
# subset with variation

# %% tags=["hide-input"]
try:
    display(meta_stats.loc[:, (meta_stats.loc['unique']
            > 1) | (meta_stats.loc['std'] > 0.1)])
except KeyError:
    if 'std' in meta_stats.index:
        display(meta_stats.loc[:, (meta_stats.loc['std'] > 0.1)])
    if 'unique' in meta_stats.index:
        display(meta_stats.loc[:, (meta_stats.loc['std'] > 0.1)])


# %% tags=["hide-input"]
df_meta = align_meta_data(df, df_meta=df_meta)

# %% [markdown]
# Ensure unique indices

# %% tags=["hide-input"]
assert df.index.is_unique, "Duplicates in index."

# %% [markdown]
# ## Select a subset of samples if specified (reduce the number of samples)
#
# - select features if `select_N` is specifed (for now)
# - for interpolation to make sense, it is best to select a consecutive number of samples:
#   - take N most recent samples (-> check that this makes sense for your case)

# %% tags=["hide-input"]
if args.select_N is not None:
    args.select_N = min(args.select_N, len(df_meta))
    if args.sample_N:
        df_meta = df_meta.sample(args.select_N)
    else:
        df_meta = df_meta.iloc[-args.select_N:]

    df = df.loc[df_meta.index].dropna(
        how='all', axis=1)
    ax = df.T.describe().loc['count'].hist()
    _ = ax.set_title('histogram of features for all eligable samples')

# %% [markdown]
# ## First Step: Select features by prevalence
# - `feat_prevalence` across samples


# %% tags=["hide-input"]
# ! add function
freq_per_feature = df.notna().sum()  # on wide format
if isinstance(args.feat_prevalence, float):
    N_samples = len(df)
    logger.info(f"Current number of samples: {N_samples}")
    logger.info(
        f"Feature has to be present in at least {args.feat_prevalence:.2%} of samples")
    args.overwrite_entry('feat_prevalence', int(
        N_samples * args.feat_prevalence))
assert isinstance(args.feat_prevalence, int)
# ! check that feature prevalence is greater equal to 3 (otherwise train, val, test split is not possible)
logger.info(
    f"Feature has to be present in at least {args.feat_prevalence} of samples")
# select features
mask = freq_per_feature >= args.feat_prevalence
logger.info(f"Drop {(~mask).sum()} features")
freq_per_feature = freq_per_feature.loc[mask]
df = df.loc[:, mask]
df

# %% tags=["hide-input"]
notna = df.notna()
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

# %% tags=["hide-input"]
if isinstance(args.sample_completeness, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {args.sample_completeness}\n'
    # assumes df in wide format
    args.overwrite_entry('sample_completeness', int(
        df.shape[1] * args.sample_completeness))
    msg += f'This translates to a minimum number of features per sample (to be included): {args.sample_completeness}'
    logger.info(msg)

sample_counts = df.notna().sum(axis=1)  # if DataFrame
sample_counts.describe()

# %% tags=["hide-input"]
mask = sample_counts > args.sample_completeness
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
logger.info(msg)
df = df.loc[mask]
df = df.dropna(
    axis=1, how='all')  # drop now missing features

# %% tags=["hide-input"]
args.N, args.M = df.shape  # save data dimensions
args.used_samples = df.index.to_list()

# %% [markdown]
# ### Histogram of features per sample

# %% tags=["hide-input"]
group = 1
ax = df.notna().sum(axis=1).hist()
ax.set_xlabel(f'{args.feat_name_display.capitalize()} per eligable sample')
ax.set_ylabel('observations')
fname = args.out_figures / f'0_{group}_hist_features_per_sample'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
ax = df.notna().sum(axis=0).sort_values().plot()
_new_labels = [l_.get_text().split(';')[0] for l_ in ax.get_xticklabels()]
_ = ax.set_xticklabels(_new_labels, rotation=45,
                       horizontalalignment='right')
ax.set_xlabel(f'{args.feat_name_display.capitalize()} prevalence')
ax.set_ylabel('observations')
fname = args.out_figures / f'0_{group}_feature_prevalence'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)


# %% [markdown]
# ### Number off observations accross feature value

# %% tags=["hide-input"]
min_max = vaep.plotting.data.min_max(df.stack())
ax, bins = vaep.plotting.data.plot_histogram_intensities(
    df.stack(), min_max=min_max)
ax.set_xlabel('Intensity binned')
fname = args.out_figures / f'0_{group}_intensity_distribution_overall'

figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='scatter')
fname = args.out_figures / f'0_{group}_intensity_median_vs_prop_missing_scatter'
ax.set_xlabel(
    f'{args.feat_name_display.capitalize()} binned by their median intensity'
    f' (N {args.feat_name_display})')
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
ax, _data_feat_median_over_prop_missing = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='boxplot', return_plot_data=True)
fname = args.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot'
ax.set_xlabel(
    f'{args.feat_name_display.capitalize()} binned by their median intensity'
    f' (N {args.feat_name_display})')
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)
_data_feat_median_over_prop_missing.to_csv(fname.with_suffix('.csv'))
# _data_feat_median_over_prop_missing.to_excel(fname.with_suffix('.xlsx'))
del _data_feat_median_over_prop_missing

# %% [markdown]
# ### Interactive and Single plots

# %% tags=["hide-input"]
_feature_display_name = f'identified {args.feat_name_display}'
sample_counts.name = _feature_display_name

# %% tags=["hide-input"]
K = 2
df = df.astype(float)
pcs = get_PCA(df, n_components=K)  # should be renamed to get_PCs
pcs = pcs.iloc[:, :K].join(df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs_index_name = pcs.index.name
pcs = pcs.reset_index()
pcs

# %% tags=["hide-input"]
pcs.describe(include='all').T

# %% tags=["hide-input"]
if args.meta_cat_col:
    fig, ax = plt.subplots(figsize=(3, 3))
    analyzers.seaborn_scatter(
        pcs[pcs_name], ax, meta=pcs[args.meta_cat_col], title=f"by {args.meta_cat_col}")
    fname = (args.out_figures
             / f'0_{group}_pca_sample_by_{"_".join(args.meta_cat_col.split())}')
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% tags=["hide-input"]
if args.meta_date_col != 'PlaceholderTime':
    fig, ax = plt.subplots()
    analyzers.plot_date_map(
        df=pcs[pcs_name], ax=ax, dates=pcs[args.meta_date_col], title=f'by {args.meta_date_col}')
    fname = args.out_figures / f'0_{group}_pca_sample_by_date'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
# - size: number of features in a single sample

# %% tags=["hide-input"]
fig, ax = plt.subplots()
col_identified_feat = _feature_display_name
analyzers.plot_scatter(
    pcs[pcs_name],
    ax,
    pcs[col_identified_feat],
    feat_name_display=args.feat_name_display,
    size=5,
)
fname = (args.out_figures
         / f'0_{group}_pca_sample_by_{"_".join(col_identified_feat.split())}.pdf')
figures[fname.stem] = fname
vaep.savefig(fig, fname)

# %% tags=["hide-input"]
# # ! write principal components to excel (if needed)
# pcs.set_index([df.index.name])[[*pcs_name, col_identified_feat]].to_excel(fname.with_suffix('.xlsx'))

# %% tags=["hide-input"]
fig = px.scatter(
    pcs, x=pcs_name[0], y=pcs_name[1],
    hover_name=pcs_index_name,
    # hover_data=analysis.df_meta,
    title=f'First two Principal Components of {args.M} {args.feat_name_display} for {pcs.shape[0]} samples',
    # color=pcs['Software Version'],
    color=col_identified_feat,
    template='none',
    width=1200,  # 4 inches x 300 dpi
    height=600  # 2 inches x 300 dpi
)
fname = (args.out_figures
         / f'0_{group}_pca_sample_by_{"_".join(col_identified_feat.split())}_plotly.pdf')
figures[fname.stem] = fname
fig.write_image(fname)
fig  # stays interactive in html

# %% [markdown]
# ## Sample Medians and percentiles

# %% tags=["hide-input"]
df.head()

# %% tags=["hide-input"]
df_w_date = df.join(df_meta[args.meta_date_col])
df_w_date = df_w_date.set_index(args.meta_date_col).sort_index()
if not args.meta_date_col == 'PlaceholderTime':
    df_w_date.to_period('min')
df_w_date = df_w_date.T
df_w_date

# %% tags=["hide-input"]
ax = df_w_date.plot.box(rot=80,
                        figsize=(7, 3),
                        fontsize=7,
                        showfliers=False,
                        showcaps=False,
                        boxprops=dict(linewidth=.4, color='darkblue'),
                        flierprops=dict(markersize=.4, color='lightblue'),
                        )
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
fname = args.out_figures / f'0_{group}_median_boxplot'
df_w_date.to_pickle(fname.with_suffix('.pkl'))
figures[fname.stem] = fname
vaep.savefig(fig, fname)
del df_w_date

# %% [markdown]
# Percentiles of intensities in dataset

# %% tags=["hide-input"]
df.stack().describe(percentiles=np.linspace(0.05, 0.95, 19).round(2))

# %% [markdown]
# ### Plot sample median over time
#   - check if points are equally spaced (probably QC samples are run in close proximity)
#   - the machine will be not use for intermediate periods

# %% tags=["hide-input"]
if not args.meta_date_col == 'PlaceholderTime':
    dates = df_meta[args.meta_date_col].sort_values()
    median_sample_intensity = (df
                               .median(axis=1)
                               .to_frame('median intensity'))
    median_sample_intensity = median_sample_intensity.join(dates)

    ax = median_sample_intensity.plot.scatter(x=dates.name, y='median intensity',
                                              rot=90,
                                              #   fontsize=6,
                                              figsize=(8, 2),
                                              s=5,
                                              xticks=vaep.plotting.select_dates(
                                                  median_sample_intensity[dates.name])
                                              )
    fig = ax.get_figure()
    fname = args.out_figures / f'0_{group}_median_scatter'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
# - the closer the labels are there denser the samples are measured around that time.

# %% [markdown]
# ## Feature frequency  in data

# %% tags=["hide-input"]
msg = "Total number of samples in data: {}"
logger.info(msg.format(len(df)))


# %% [markdown]
# Recalculate feature frequency after selecting samples

# %% tags=["hide-input"]
freq_per_feature = feature_frequency(df)
freq_per_feature

# %% tags=["hide-input"]
# freq_per_feature.name = 'Gene names freq' # name it differently?
# index.name is lost when data is stored
fname = args.data / 'freq_features.json'
dumps[fname.name] = fname
freq_per_feature.to_json(fname)
fname = fname.with_suffix('.pkl')
dumps[fname.name] = fname
freq_per_feature.to_pickle(fname)


# %% [markdown]
# ## Split: Train, validation and test data
#
# Select features as described in
# > Lazar, Cosmin, Laurent Gatto, Myriam Ferro, Christophe Bruley, and Thomas Burger. 2016.
# > “Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative
# > Proteomics Data Sets to Compare Imputation Strategies.”
# > Journal of Proteome Research 15 (4): 1116–25.
#
# - select `frac_mnar` based on threshold matrix on quantile of overall frac of data to be used
#   for validation and test data split, e.g. 0.1 = quantile(0.1)
# - select frac_mnar from intensities selected using threshold matrix

# %% tags=["hide-input"]
splits = DataSplits(is_wide_format=False)
logger.info(f"{splits = }")
splits.__annotations__


# %% [markdown]
# Create some target values by sampling X% of the validation and test data.
# Simulated missing values are not used for validation and testing.

# %% tags=["hide-input"]
df_long = vaep.io.datasplits.long_format(df)
df_long.head()

# %% tags=["hide-input"]
group = 2

splits, thresholds, fake_na_mcar, fake_na_mnar = vaep.sampling.sample_mnar_mcar(
    df_long=df_long,
    frac_non_train=args.frac_non_train,
    frac_mnar=args.frac_mnar,
    random_state=args.random_state,
)
logger.info(f"{splits.train_X.shape = } - {splits.val_y.shape = } - {splits.test_y.shape = }")

# %% tags=["hide-input"]
N = len(df_long)
N_MCAR = len(fake_na_mcar)
N_MNAR = len(fake_na_mnar)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))
ax = axes[0]
plot_histogram_intensities = partial(vaep.plotting.data.plot_histogram_intensities,
                                     min_max=min_max,
                                     alpha=0.8)
plot_histogram_intensities(
    df_long.squeeze(),
    ax=ax,
    label='observed')
plot_histogram_intensities(
    thresholds,
    ax=ax,
    label='thresholds')
if args.use_every_nth_xtick > 1:
    ax.set_xticks(ax.get_xticks()[::2])
ax.legend()
ax = axes[1]
plot_histogram_intensities(
    fake_na_mnar.squeeze(),
    ax=ax,
    label=f'MNAR ({N_MNAR:,d})',
    color='C2')
plot_histogram_intensities(
    fake_na_mcar.squeeze(),
    ax=ax,
    color='C3',
    label=f'MCAR ({N_MCAR:,d})')
if args.use_every_nth_xtick > 1:
    ax.set_xticks(ax.get_xticks()[::2])
ax.legend()
fname = args.out_figures / f'0_{group}_mnar_mcar_histograms.pdf'
figures[fname.stem] = fname
vaep.savefig(fig, fname)

# %% tags=["hide-input"]
counts_per_bin = vaep.pandas.get_counts_per_bin(
    df=pd.concat(
        [df_long.squeeze().to_frame('observed'),
         thresholds.to_frame('threshold'),
         fake_na_mnar.squeeze().to_frame(f'MNAR ({N_MNAR:,d})'),
         fake_na_mcar.squeeze().to_frame(f'MCAR ({N_MCAR:,d})')],
        axis=1),
    bins=range(min_max[0], min_max[1] + 1, 1))
counts_per_bin.to_excel(fname.with_suffix('.xlsx'))
counts_per_bin


# %% [markdown]
# ### Keep simulated samples only in a subset of the samples
# In case only a subset of the samples should be used for validation and testing,
# although these samples can be used for fitting the models,
# the following cell will select samples stratified by the eventually set `meta_cat_col` column.
#
# The procedure is experimental and turned off by default.

# %% tags=["hide-input"]
if 0.0 < args.prop_sample_w_sim < 1.0:
    to_stratify = None
    if args.meta_cat_col and df_meta is not None:
        to_stratify = df_meta[args.meta_cat_col].fillna(-1)  # ! fillna with -1 as separate category (sofisticate check)
    train_idx, val_test_idx = train_test_split(splits.train_X.index.levels[0],
                                               test_size=args.prop_sample_w_sim,
                                               stratify=to_stratify,
                                               random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx,
                                         test_size=.5,
                                         stratify=to_stratify.loc[val_test_idx] if to_stratify is not None else None,
                                         random_state=42)
    logger.info(f"Sample in Train: {len(train_idx):,d} - Validation: {len(val_idx):,d} - Test: {len(test_idx):,d}")
    # reassign some simulated missing values to training data:
    splits.train_X = pd.concat(
        [splits.train_X,
         splits.val_y.loc[train_idx],
         splits.test_y.loc[train_idx]
         ])
    splits.val_y = splits.val_y.loc[val_idx]
    splits.test_y = splits.test_y.loc[test_idx]
    logger.info(f"New shapes: {splits.train_X.shape = } - {splits.val_y.shape = } - {splits.test_y.shape = }")

# %% tags=["hide-input"]
splits.test_y.groupby(level=-1).count().describe()

# %% tags=["hide-input"]
splits.val_y

# %% tags=["hide-input"]
splits.train_X.groupby(level=-1).count().describe()

# %% tags=["hide-input"]
# Check that feature indices and sample indicies overlap between splits
# -> a single feature cannot be only in the validation or test split
# -> single features should be put into the training data
# -> or raise error as feature completness treshold is so low that less than 3 samples
# per feature are allowd.

splits = vaep.sampling.check_split_integrity(splits)

# %% [markdown]
# Some tools require at least 4 observation in the training data,
# which is a good requirment. Due to "MNAR" sampling, most measurments
# of a features can end up in the validation or test data.
#
# In that case: Move the validation measurments back to the training data.
# If after this procedure the condition is still not met, a value error is raised.

# %% tags=["hide-input"]
mask_min_4_measurments = splits.train_X.groupby(level=1).count() < 4
if mask_min_4_measurments.any():
    idx = mask_min_4_measurments.loc[mask_min_4_measurments].index
    logger.warning(f"Features with less than 4 measurments in training data: {idx.to_list()}")
    to_remove = splits.val_y.loc[pd.IndexSlice[:, idx]]
    logger.info("To remove from validation data: ")
    display(to_remove)
    splits.train_X = pd.concat([splits.train_X, to_remove])
    splits.val_y = splits.val_y.drop(to_remove.index)
    # check condition again
    mask_min_4_measurments = splits.train_X.groupby(level=1).count() < 4
    if mask_min_4_measurments.any():
        idx = mask_min_4_measurments.loc[mask_min_4_measurments].index
        raise ValueError("Some features still have less than 4 measurments in training data"
                         f" after removing the features from the validation data: {idx.to_list()}")

# %% [markdown]
# ### Save in long format
#
# - Data in long format: (peptide, sample_id, intensity)
# - no missing values kept

# %% tags=["hide-input"]
# dumps data in long-format
splits_dumped = splits.dump(folder=args.data, file_format=args.file_format)
dumps.update(splits_dumped)
splits_dumped

# %% [markdown]
# ### Reload from disk

# %% tags=["hide-input"]
splits = DataSplits.from_folder(args.data, file_format=args.file_format)

# %% [markdown]
# ## plot distribution of splits

# %% tags=["hide-input"]
splits_df = pd.DataFrame(index=df_long.index)
splits_df['train'] = splits.train_X
splits_df['val'] = splits.val_y
splits_df['test'] = splits.test_y
stats_splits = splits_df.describe()
stats_splits.to_excel(writer, 'stats_splits', float_format='%.3f')
stats_splits

# %% tags=["hide-input"]
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

# %% tags=["hide-input"]
group = 3
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
if args.use_every_nth_xtick > 1:
    ax.set_xticks(ax.get_xticks()[::2])
ax.set_xlabel('Intensity bins')
fname = args.out_figures / f'0_{group}_val_over_train_split.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
min_bin, max_bin = vaep.plotting.data.min_max(splits.val_y)
bins = range(int(min_bin), int(max_bin) + 1, 1)
ax = splits_df.plot.hist(bins=bins,
                         xticks=list(bins),
                         legend=False,
                         stacked=True,
                         color=['C0', 'C1', 'C2'],
                         )
if args.use_every_nth_xtick > 1:
    ax.set_xticks(ax.get_xticks()[::2])
ax.legend(_legend)
ax.set_xlabel('Intensity bins')
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = args.out_figures / f'0_{group}_splits_freq_stacked.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
counts_per_bin = vaep.pandas.get_counts_per_bin(df=splits_df, bins=bins)
counts_per_bin.to_excel(fname.with_suffix('.xlsx'))
counts_per_bin

# %% tags=["hide-input"]
ax = splits_df.drop('train', axis=1).plot.hist(bins=bins,
                                               xticks=list(bins),
                                               color=['C1', 'C2'],
                                               legend=False,
                                               stacked=True,
                                               )
if args.use_every_nth_xtick > 1:
    ax.set_xticks(ax.get_xticks()[::2])
ax.legend(_legend[1:])
ax.set_xlabel('Intensity bins')
ax.yaxis.set_major_formatter("{x:,.0f}")
fname = args.out_figures / f'0_{group}_val_test_split_freq_stacked_.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)


# %% [markdown]
# plot training data missing plots

# %% tags=["hide-input"]
splits.to_wide_format()

# %% tags=["hide-input"]
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='scatter')
fname = args.out_figures / f'0_{group}_intensity_median_vs_prop_missing_scatter_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='boxplot')
fname = args.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% tags=["hide-input"]
medians = (splits
           .train_X
           .median()
           .astype(int)
           ).to_frame('median_floor')

feat_with_median = medians.groupby('median_floor').size().rename('M feat')
medians = medians.join(feat_with_median, on='median_floor')
medians = medians.apply(lambda s: "{:02,d} (N={:3,d})".format(*s), axis=1)

fig, ax = plt.subplots(figsize=(6, 2))
s = 1
s_axes = pd.DataFrame({'medians': medians,
                       'Validation split': splits.val_y.notna().sum(),
                       'Training split': splits.train_X.notna().sum()}
                      ).plot.box(by='medians',
                                 boxprops=dict(linewidth=s),
                                 flierprops=dict(markersize=s),
                                 ax=ax)
for ax in s_axes:
    _ = ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           horizontalalignment='right')
    ax.set_xlabel(f'{args.feat_name_display.capitalize()} binned by their median intensity '
                  f'(N {args.feat_name_display})')
    _ = ax.set_ylabel('Frequency')
fname = args.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot_val_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ## Save parameters

# %% tags=["hide-input"]
fname = args.folder_experiment / 'data_config.yaml'
args.dump(fname)
args

# %% [markdown]
# ## Saved Figures

# %% tags=["hide-input"]
# saved figures
figures

# %% [markdown]
# Saved dumps

# %% tags=["hide-input"]
writer.close()
dumps
