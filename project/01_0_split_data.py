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

# %%
from pathlib import Path
import logging
from typing import Union, List

from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

import vaep
from vaep.io.datasplits import DataSplits
from vaep.sampling import feature_frequency, sample_data

from vaep.analyzers import analyzers
from vaep.analyzers.analyzers import AnalyzePeptides

logger = vaep.logging.setup_nb_logger()
logger.info("Split data and make diagnostic plots")
logging.getLogger('fontTools').setLevel(logging.WARNING)


def add_meta_data(df: pd.DataFrame, df_meta: pd.DataFrame):
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

vaep.plotting.make_large_descriptors(6)

figures = {}  # collection of ax or figures
dumps = {}  # collection of data dumps

# %% [markdown]
# ## Arguments

# %%
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
min_RT_time: Union[int, float] = None  # based on raw file meta data, only take samples with RT > min_RT_time
logarithm: str = 'log2'  # Log transformation of initial data (select one of the existing in numpy)
folder_experiment: str = 'runs/example'  # folder to save figures and data dumps
folder_data: str = ''  # specify special data directory if needed
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
# metadata -> defaults for metadata extracted from machine data, used for plotting
meta_date_col: str = None  # date column in meta data
meta_cat_col: str = None  # category column in meta data
# train, validation and test data splits
frac_non_train: float = 0.1  # fraction of non training data (validation and test split)
frac_mnar: float = 0.0  # fraction of missing not at random data, rest: missing completely at random


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
# # ! factor out file reading to a separate module, not class
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
df = analysis.df
del analysis.df  # free memory
df

# %%
ax = (df
      .notna()
      .sum(axis=0)
      .to_frame(df.columns.name)
      .plot
      .box()
      )
ax.set_ylabel('number of observation across samples')


# %%
fname = params.out_folder / '01_0_data_stats.xlsx'
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
# In case there are multiple features for each intensity values (currenlty: peptide sequence and charge), combine the column names to a single str index.
#
# > The Collaborative Modeling approach will need a single feature column.

# %%
def join_as_str(seq):
    ret = "_".join(str(x) for x in seq)
    return ret


# ToDo: join multiindex samples indices (pkl dumps)
# if hasattr(df.columns, "levels"):
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
    df_meta = pd.DataFrame(index=df.index)
df_meta = df_meta.loc[df.index.to_list()]  # index is sample index
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
df_meta.describe(percentiles=np.linspace(0.05, 0.95, 10))

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
    logger.warning("Retention time filtering deactivated.")

# %%
df_meta = df_meta.sort_values(params.meta_date_col)

# %%
meta_stats = df_meta.describe(include='all')
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


# %%
df_meta = add_meta_data(df, df_meta=df_meta)

# %% [markdown]
# Ensure unique indices

# %%
assert df.index.is_unique, "Duplicates in index"

# %% [markdown]
# ## Select a subset of samples if specified (reduce the number of samples)
#
# - select features if `select_N` is specifed (for now)
# - for interpolation to make sense, it is best to select a consecutive number of samples:
#   - take N most recent samples (-> check that this makes sense for your case)

# %%
if params.select_N is not None:
    params.select_N = min(params.select_N, len(df_meta))
    if params.sample_N:
        df_meta = df_meta.sample(params.select_N)
    else:
        df_meta = df_meta.iloc[-params.select_N:]

    df = df.loc[df_meta.index].dropna(
        how='all', axis=1)
    ax = df.T.describe().loc['count'].hist()
    _ = ax.set_title('histogram of features for all eligable samples')

# %% [markdown]
# ## First Step: Select features by prevalence
# - `feat_prevalence` across samples


# %%
freq_per_feature = df.notna().sum()  # on wide format
if isinstance(params.feat_prevalence, float):
    N_samples = len(df)
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
df = df.loc[:, mask]
analysis.N, analysis.M = df.shape
# # potentially create freq based on DataFrame
df

# %%
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

# %%
if isinstance(params.sample_completeness, float):
    msg = f'Fraction of minimum sample completeness over all features specified with: {params.sample_completeness}\n'
    # assumes df in wide format
    params.overwrite_entry('sample_completeness', int(
        df.shape[1] * params.sample_completeness))
    msg += f'This translates to a minimum number of features per sample (to be included): {params.sample_completeness}'
    logger.info(msg)

sample_counts = df.notna().sum(axis=1)  # if DataFrame
sample_counts.describe()

# %%
mask = sample_counts > params.sample_completeness
msg = f'Drop {len(mask) - mask.sum()} of {len(mask)} initial samples.'
print(msg)
df = df.loc[mask]
df = df.dropna(
    axis=1, how='all')  # drop now missing features

# %%
params.N, params.M = df.shape  # save data dimensions
params.used_samples = df.index.to_list()

# %% [markdown]
# ### Histogram of features per sample

# %%
group = 1
ax = df.notna().sum(axis=1).hist()
ax.set_xlabel('features per eligable sample')
ax.set_ylabel('observations')
fname = params.out_figures / f'0_{group}_hist_features_per_sample'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = df.notna().sum(axis=0).sort_values().plot()
_new_labels = [l.get_text().split(';')[0] for l in ax.get_xticklabels()]
_ = ax.set_xticklabels(_new_labels, rotation=45,
                       horizontalalignment='right')
ax.set_xlabel('feature prevalence')
ax.set_ylabel('observations')
fname = params.out_figures / f'0_{group}_feature_prevalence'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)


# %% [markdown]
# ### Number off observations accross feature value

# %%
min_max = vaep.plotting.data.min_max(df.stack())
ax, bins = vaep.plotting.data.plot_histogram_intensities(
    df.stack(), min_max=min_max)

fname = params.out_figures / f'0_{group}_intensity_distribution_overall'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='scatter')
fname = params.out_figures / f'0_{group}_intensity_median_vs_prop_missing_scatter'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=df, type='boxplot')
fname = params.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# ### Interactive and Single plots

# %%
sample_counts.name = 'identified features'

# %%
K = 2
df = df.astype(float)
analysis.df = df
pcs = analysis.get_PCA(n_components=K)  # should be renamed to get_PCs
pcs = pcs.iloc[:, :K].join(df_meta).join(sample_counts)

pcs_name = pcs.columns[:2]
pcs_index_name = pcs.index.name
pcs = pcs.reset_index()
pcs

# %%
pcs.describe(include='all').T

# %%
if params.meta_cat_col:
    fig, ax = plt.subplots(figsize=(2, 2))
    analyzers.seaborn_scatter(
        pcs[pcs_name], ax, meta=pcs[params.meta_cat_col], title=f"by {params.meta_cat_col}")
    fname = (params.out_figures
             / f'0_{group}_pca_sample_by_{"_".join(params.meta_cat_col.split())}')
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %%
if params.meta_date_col != 'PlaceholderTime':
    fig, ax = plt.subplots()
    analyzers.plot_date_map(
        df=pcs[pcs_name], ax=ax, dates=pcs[params.meta_date_col], title=f'by {params.meta_date_col}')
    fname = params.out_figures / f'0_{group}_pca_sample_by_date'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
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
         / f'0_{group}_pca_sample_by_{"_".join(col_identified_feat.split())}.pdf')
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
    width=1200,  # 4 inches x 300 dpi
    height=600  # 2 inches x 300 dpi
)
fname = (params.out_figures
         / f'0_{group}_pca_sample_by_{"_".join(col_identified_feat.split())}_plotly.pdf')
figures[fname.stem] = fname
fig.write_image(fname)
fig  # stays interactive in html

# %% [markdown]
# ## Sample Medians and percentiles

# %%
df.head()

# %%
df_w_date = df.join(df_meta[params.meta_date_col])
df_w_date = df_w_date.set_index(params.meta_date_col).sort_index()
if not params.meta_date_col == 'PlaceholderTime':
    df_w_date.to_period('min')
df_w_date = df_w_date.T
df_w_date

# %%
ax = df_w_date.boxplot(rot=80,
                       figsize=(8, 3),
                       fontsize=6,
                       showfliers=False,
                       showcaps=False
                       )
_ = vaep.plotting.select_xticks(ax)
fig = ax.get_figure()
fname = params.out_figures / f'0_{group}_median_boxplot'
figures[fname.stem] = fname
vaep.savefig(fig, fname)
del df_w_date

# %% [markdown]
# Percentiles of intensities in dataset

# %%
df.stack().describe(percentiles=np.linspace(0.05, 0.95, 19).round(2))

# %% [markdown]
# ### Plot sample median over time
#   - check if points are equally spaced (probably QC samples are run in close proximity)
#   - the machine will be not use for intermediate periods

# %%
if not params.meta_date_col == 'PlaceholderTime':
    dates = df_meta[params.meta_date_col].sort_values()
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
    fname = params.out_figures / f'0_{group}_median_scatter'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)

# %% [markdown]
# - the closer the labels are there denser the samples are measured around that time.

# %% [markdown]
# ## Feature frequency  in data

# %%
msg = "Total number of samples in data: {}"
print(msg.format(len(df)))


# %% [markdown]
# Recalculate feature frequency after selecting samples

# %%
freq_per_feature = feature_frequency(df)
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

# %%
splits = DataSplits(is_wide_format=False)
print(f"{splits = }")
splits.__annotations__


# %% [markdown]
# Create some target values by sampling X% of the validation and test data.
# Simulated missing values are not used for validation and testing.

# %%
df_long = vaep.io.datasplits.long_format(df)
df_long.head()

# %%
group = 2
# if not mnar:
#     fake_na, splits.train_X = sample_data(df_long.squeeze(),
#                                           sample_index_to_drop=0,
#                                           weights=freq_per_feature,
#                                           frac=0.1,
#                                           random_state=params.random_state,)
#     assert len(splits.train_X) > len(fake_na)
# ! move parameter checks to start of script
if 0.0 <= params.frac_mnar <= 1.0:
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    quantile_frac = df_long.quantile(params.frac_non_train)
    rng = np.random.default_rng(params.random_state)
    threshold = pd.Series(rng.normal(loc=float(quantile_frac),
                                     scale=float(0.3 * df_long.std()),
                                     size=len(df_long),
                                     ),
                          index=df_long.index,
                          )
    # plot data vs threshold data
    ax = axes[0]
    from functools import partial
    plot_histogram_intensities = partial(vaep.plotting.data.plot_histogram_intensities,
                                         min_max=min_max,
                                         alpha=0.8)
    plot_histogram_intensities(
        df_long.squeeze(),
        ax=ax,
        label='observed')
    plot_histogram_intensities(
        threshold,
        ax=ax,
        label='thresholds')
    ax.legend()
    # select MNAR (intensity between randomly sampled threshold)
    mask = df_long.squeeze() < threshold
    # ! subsample to have exact fraction of MNAR?
    N = len(df_long)
    logger.info(f"{int(N * params.frac_non_train) = :,d}")
    N_MNAR = int(params.frac_non_train * params.frac_mnar * N)
    fake_na_mnar = df_long.loc[mask]
    if len(fake_na_mnar) > N_MNAR:
        fake_na_mnar = fake_na_mnar.sample(N_MNAR,
                                           random_state=params.random_state)
    splits.train_X = df_long.loc[
        df_long.index.difference(
            fake_na_mnar.index)
    ]
    logger.info(f"{len(fake_na_mnar) = :,d}")
    N_MCAR = int(N * (1 - params.frac_mnar) * params.frac_non_train)
    fake_na_mcar = splits.train_X.sample(N_MCAR,
                                         random_state=params.random_state)
    logger.info(f"{len(fake_na_mcar) = :,d}")
    splits.train_X = (splits
                      .train_X
                      .loc[splits
                           .train_X
                           .index
                           .difference(
                               fake_na_mcar.index)]
                      ).squeeze()
    logger.info(f"{len(splits.train_X) = :,d}")
    fake_na = pd.concat([fake_na_mcar, fake_na_mnar]).squeeze()
    logger.info(f"{len(fake_na) = :,d}")
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
    ax.legend()
    assert len(fake_na) + len(splits.train_X) == len(df_long)
    fname = params.out_figures / f'0_{group}_mnar_mcar_histograms.pdf'
    figures[fname.stem] = fname
    vaep.savefig(fig, fname)
else:
    raise ValueError(f"Invalid MNAR float value (should be betw. 0 and 1): {params.frac_mnar}")

splits.val_y = fake_na.sample(frac=0.5, random_state=params.random_state)
splits.test_y = fake_na.loc[fake_na.index.difference(splits.val_y.index)]

# %%
splits.test_y.groupby(level=-1).count().describe()

# %%
splits.val_y

# %%
# ! add option to retain at least N samples per feature
splits.train_X.groupby(level=-1).count().describe()

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
splits_df = pd.DataFrame(index=df_long.index)
splits_df['train'] = splits.train_X
splits_df['val'] = splits.val_y
splits_df['test'] = splits.test_y
stats_splits = splits_df.describe()
# stats_splits.to_excel(writer, 'stats_splits', float_format='%.2f')
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
fname = params.out_figures / f'0_{group}_test_over_train_split.pdf'
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
fname = params.out_figures / f'0_{group}_splits_freq_stacked.pdf'
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
fname = params.out_figures / f'0_{group}_val_test_split_freq_stacked_.pdf'
figures[fname.name] = fname
vaep.savefig(ax.get_figure(), fname)

# %% [markdown]
# plot training data missing plots

# %%
splits.to_wide_format()

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='scatter')
fname = params.out_figures / f'0_{group}_intensity_median_vs_prop_missing_scatter_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
ax = vaep.plotting.data.plot_feat_median_over_prop_missing(
    data=splits.train_X, type='boxplot')
fname = params.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot_train'
figures[fname.stem] = fname
vaep.savefig(ax.get_figure(), fname)

# %%
medians = (splits
           .train_X
           .median()
           .astype(int)
           ).to_frame('median_floor')

feat_with_median = medians.groupby('median_floor').size().rename('M feat')
medians = medians.join(feat_with_median, on='median_floor')
medians = medians.apply(lambda s: "{:02,d} (N={:3,d})".format(*s), axis=1)

fig, ax = plt.subplots(figsize=(8, 2))
s = 1
s_axes = pd.DataFrame({'medians': medians,
                       'validation split': splits.val_y.notna().sum(),
                       'training split': splits.train_X.notna().sum()}
                      ).plot.box(by='medians',
                                 boxprops=dict(linewidth=s),
                                 flierprops=dict(markersize=s),
                                 ax=ax)
for ax in s_axes:
    _ = ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           horizontalalignment='right')

fname = params.out_figures / f'0_{group}_intensity_median_vs_prop_missing_boxplot_val_train'
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
