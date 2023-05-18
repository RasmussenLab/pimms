# ---
# jupyter:
#   jupytext:
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
# # Split up data into single datasets
#
# - create datasets per (set of) instruments for a specific experiments
# - drop some samples based on quality criteria

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns

import umap

from vaep.io import thermo_raw_files
from vaep.analyzers import analyzers

from config import erda_dumps
from config import defaults

import vaep
import vaep.io.filenames
from vaep.logging import setup_nb_logger

logger = setup_nb_logger()

FOLDER_DATA = defaults.FOLDER_DATA

# %%
vaep.plotting.make_large_descriptors()
FIGSIZE = (15, 10)

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
N_MIN_INSTRUMENT = 300
META_DATA: str = 'data/files_selected_metadata.csv'
FILE_EXT = 'pkl' # 'csv' or 'pkl'
SAMPLE_ID = 'Sample ID'

DUMP: str = erda_dumps.FN_PROTEIN_GROUPS # Filepath to erda dump
OUT_NAME = 'protein group'  # for legends labels
# DUMP: str = erda_dumps.FN_PEPTIDES
# OUT_NAME = 'peptide' # for legends labels
# DUMP: str = erda_dumps.FN_EVIDENCE
# OUT_NAME = 'precursor' # for legends labels

FOLDER_DATASETS: str = f'dev_datasets/{DUMP.stem}'

INSTRUMENT_LEGEND_TITLE = 'Q Exactive HF-X Orbitrap'

# %%
# FILE_EXT = 'csv'

# %% [markdown]
# Make sure output folder exists

# %%
DUMP = Path(DUMP)  # set parameter from cli or yaml to Path
FOLDER_DATASETS = defaults.FOLDER_DATA / FOLDER_DATASETS
FOLDER_DATASETS.mkdir(exist_ok=True, parents=True)
logger.info(f"Folder for datasets to be created: {FOLDER_DATASETS.absolute()}")

files_out = dict()

# %% [markdown]
# ## Dumps

# %% [markdown]
# - load dumps
# - load file to machine mappings

# %%
data = pd.read_pickle(DUMP)
data = data.squeeze()  # In case it is a DataFrame, not a series (-> leads to MultiIndex)
# name_data = data.name
logger.info(
    f"Number of rows (row = sample, feature, intensity): {len(data):,d}")
data

# %% [markdown]
# Make categorical index a normal string index (this lead to problems when selecting data using `loc` and grouping data as level of data could not easily be removed from MultiIndex)
#
# - see [blog](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a)

# %%
# index_columns = data.index.names
# data = data.reset_index()
# print(data.memory_usage(deep=True))
# cat_columns = data.columns[data.dtypes == 'category']
# if not cat_columns.empty:
#     data[cat_columns] = data[cat_columns].astype('object')
#     print("non categorical: \n", data.memory_usage(deep=True))
#     logger.warning(
#         "if time allows, this should be investigate -> use of loc with data which is not categorical")
# data = data.set_index(index_columns)

# %%
# feat_name = list(data.index.names)
# feat_name.remove(SAMPLE_ID)
feat_name = (OUT_NAME,)
feat_name # index name(s) which are not the sample index

# %%
# M = len(data.index.levels[-1])
N, M = data.shape
logger.info(f"Number of unqiue features: {M}")

# %% [markdown]
# ## Filter data by metadata

# %%
# sample_ids = data.index.levels[0] # assume first index position is Sample ID?
sample_ids = data.index.unique() #.get_level_values(SAMPLE_ID).unique()  # more explict
sample_ids

# %% [markdown]
# ### Meta Data
#
# - based on ThermoRawFileParser
# %%
df_meta = pd.read_csv(META_DATA, index_col=SAMPLE_ID)
date_col = 'Content Creation Date'
df_meta[date_col] = pd.to_datetime(df_meta[date_col])
df_meta = df_meta.loc[sample_ids]
df_meta

# %% [markdown]
# ## Rename samples
# - to "YEAR_MONTH_DAY_HOUR_MIN_INSTRUMENT" (no encoding of information intended)
# - check that instrument names are unique
# - drop metadata (entire)
# %%
idx_all = (pd.to_datetime(df_meta["Content Creation Date"]).dt.strftime("%Y_%m_%d_%H_%M")
        + '_'
        + df_meta["Thermo Scientific instrument model"].str.replace(' ', '-')
        + '_'
        + df_meta["instrument serial number"].str.split('#').str[-1])

mask = idx_all.duplicated(keep=False)
duplicated_sample_idx = idx_all.loc[mask].sort_values()  # duplicated dumps
duplicated_sample_idx

#
# %%
data_duplicates = data.loc[duplicated_sample_idx.index] #.unstack()
# data_duplicates.T.corr() # same samples are have corr. of 1
data_duplicates.sum(axis=1) # keep only one seems okay

# %%
idx_unique = idx_all.drop_duplicates()
idx_unique

# %%
df_meta = df_meta.loc[idx_unique.index].rename(idx_unique)
df_meta

# %%
# data = data.unstack(feat_name) # needed later anyways
data = data.loc[idx_unique.index].rename(idx_unique)
data

# %%
meta_to_drop = ['Pathname']
fname = FOLDER_DATASETS / 'metadata.csv'
files_out[fname.name] = fname
df_meta.drop(meta_to_drop, axis=1).to_csv(fname)
logger.info(f"{fname = }")


# %% [markdown]
# ## Support per sample in entire data set

# %%
counts = data.count(axis=1) # wide format
N = len(counts)
fname = FOLDER_DATASETS / 'support_all.json'
files_out[fname.name] = fname
counts.to_json(fname, indent=4)
ax = (counts
      .sort_values()  # will raise an error with a DataFrame
      .reset_index(drop=True)
      .plot(rot=45,
            figsize=FIGSIZE,
            grid=True,
            ylabel='number of features in sample',
            xlabel='Sample rank ordered by number of features',
            title=f'Support of {N:,d} samples features over {M} features ({", ".join(feat_name)})',
            ))
vaep.plotting.add_prop_as_second_yaxis(ax, M)
fig = ax.get_figure()
fig.tight_layout()
fname = FOLDER_DATASETS / 'support_all.pdf'
files_out[fname.name] = fname
vaep.plotting.savefig(fig, fname)


# %%
counts = data.count(axis=0) # wide format
counts.to_json(FOLDER_DATASETS / 'feat_completeness_all.json', indent=4)
ax = (counts
      .sort_values()  # will raise an error with a DataFrame
      .reset_index(drop=True)
      .plot(rot=45,
            figsize=FIGSIZE,
            grid=True,
            ylabel='number of samples per feature',
            xlabel='Feature rank ordered by number of samples',
            title=f'Support of {len(counts):,d} features over {N} samples ({", ".join(feat_name)})',
            ))
vaep.plotting.add_prop_as_second_yaxis(ax, N)
fig = ax.get_figure()
fname = FOLDER_DATASETS / 'feat_per_sample_all.pdf'
files_out[fname.stem] = fname
vaep.plotting.savefig(fig, fname)


# %% [markdown]
# ## Available instruments

# %%
counts_instrument = df_meta.groupby(thermo_raw_files.cols_instrument)[date_col].agg(
    ['count', 'min', 'max']).sort_values(by=thermo_raw_files.cols_instrument[:2] + ['count'], ascending=False)
counts_instrument

# %%
len(counts_instrument)

# %%
selected_instruments = counts_instrument.query(f"count >= {N_MIN_INSTRUMENT}")
fname = FOLDER_DATASETS / 'dataset_info.xlsx'
files_out[fname.name] = fname
selected_instruments.to_latex(fname.with_suffix('.tex'))
selected_instruments.to_excel(fname)
logger.info(f"Save Information to: {fname} (as xlsx and tex)")
selected_instruments


# %% [markdown]
# ## Summary plot - UMAP
#
# - embedding based on all samples
# - visualization of top 5 instruments

# %%
reducer = umap.UMAP(random_state=42)
data

# %%
embedding = reducer.fit_transform(data.fillna(data.median()))
embedding = pd.DataFrame(embedding, index=data.index,
                         columns=['UMAP 1', 'UMAP 2'])
embedding = embedding.join(
    df_meta[["Content Creation Date", "instrument serial number"]])
d_instrument_counts = counts_instrument['count'].reset_index(
    level=[0, 1], drop=True).to_dict()
embedding["count"] = embedding["instrument serial number"].replace(
    d_instrument_counts)
embedding

# %%
digits = int(np.ceil(np.log10(embedding["count"].max())))
digits

# %%
embedding["instrument with N"] = embedding[["instrument serial number",
                                            "count"]].apply(lambda s: f"{s[0]} (N={s[1]:{digits}d})", axis=1)
embedding["instrument with N"] = embedding["instrument with N"].str.replace(
    'Exactive Series slot', 'Instrument')
embedding

# %% [markdown]
# define top five instruments

# %%
top_5 = counts_instrument["count"].nlargest(5)
top_5 = top_5.index.levels[-1]
embedding["instrument"] = embedding["instrument serial number"].apply(
    lambda x: x if x in top_5 else 'other')
mask_top_5 = embedding["instrument"] != 'other'

# %%
embedding["Date (90 days intervals)"] = embedding["Content Creation Date"].dt.round(
    "90D").astype(str)
to_plot = embedding.loc[mask_top_5]
print(f"N samples in plot: {len(to_plot):,d}")
fig, ax = plt.subplots(figsize=(20, 10))

ax = sns.scatterplot(data=to_plot, x='UMAP 1', y='UMAP 2', style="instrument with N",
                     hue="Date (90 days intervals)", ax=ax)  # ="Content Creation Date")

fname = FOLDER_DATASETS / 'umap_interval90days_top5_instruments.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %%
markers = ['o', 'x', 's', 'P', 'D', '.']
alpha = 0.6
fig, ax = plt.subplots(figsize=(12, 8))
groups = list()

vaep.plotting.make_large_descriptors()
embedding["Content Creation Date"] = embedding["Content Creation Date"].dt.round(
    "D")
embedding["mdate"] = embedding["Content Creation Date"].apply(
    matplotlib.dates.date2num)

to_plot = embedding.loc[mask_top_5]

norm = matplotlib.colors.Normalize(
    embedding["mdate"].quantile(0.05), embedding["mdate"].quantile(0.95))
cmap = sns.color_palette("cubehelix", as_cmap=True)


for k, _to_plot in to_plot.groupby('instrument with N'):
    if markers:
        marker = markers.pop(0)
    _ = ax.scatter(
        x=_to_plot["UMAP 1"],
        y=_to_plot["UMAP 2"],
        c=_to_plot["mdate"],
        alpha=alpha,
        marker=marker,
        cmap=cmap,
        norm=norm
    )
    groups.append(k)

cbar = vaep.analyzers.analyzers.add_date_colorbar(
    ax.collections[0], ax=ax, fig=fig)
cbar.ax.set_ylabel("date of measurement", labelpad=-115, loc='center')
ax.legend(ax.collections, groups,
          title=INSTRUMENT_LEGEND_TITLE, fontsize='xx-large')
ax.set_xlabel('UMAP 1')  # , fontdict={'size': 16})
ax.set_ylabel('UMAP 2')

fname = FOLDER_DATASETS / 'umap_date_top5_instruments.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)

# %% [markdown]
# ## Summary statistics for top 5 instruments 

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# boxplot: number of available sample for included features
to_plot = (data
           .loc[mask_top_5]
           .notna()
           .sum(axis=0)
           .reset_index(drop=True)
           .to_frame(f'{OUT_NAME.capitalize()} prevalence')
           )
# boxplot: number of features per sample
to_plot = (to_plot
           .join(data
                 .loc[mask_top_5]
                 .notna()
                 .sum(axis=1)
                 .reset_index(drop=True)
                 .to_frame(f'{OUT_NAME.capitalize()}s per sample'))
           )
to_plot = (to_plot
           .join(counts_instrument
                 .reset_index([0, 1], drop=True)
                 .loc[top_5, 'count']
                 .reset_index(drop=True)
                 .rename('Samples per instrument', axis='index'))
           )
ax = to_plot.plot(kind='box', ax=ax, fontsize=16, )
ax.set_ylabel('number of observations',
              fontdict={'fontsize': 14})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   horizontalalignment='right')
to_plot.to_csv(FOLDER_DATASETS / 'summary_statistics_dump_data.csv')

fname = FOLDER_DATASETS / 'summary_statistics_dump.pdf'
files_out[fname.name] = fname
vaep.savefig(fig, name=fname)


# %%
top_5_meta = df_meta.loc[mask_top_5] 
top_5_meta[['injection volume setting', 'dilution factor']].describe()

# %% [markdown]
# ### Meta data stats for top 5

# %%
for _instrument, _df_meta_instrument in top_5_meta.groupby(by=thermo_raw_files.cols_instrument):
    print('#'* 80, ' - '.join(_instrument), sep='\n')
    display(_df_meta_instrument.describe())
    display(_df_meta_instrument['injection volume setting'].value_counts())
    break

# %% [markdown]
# ## Dump single experiments
#
# in wide format

# %%
# data = data.stack(feat_name)
data

# %%
cols = selected_instruments.index.names

file_formats = {'pkl': 'to_pickle',
                'pickle': 'to_pickle',
                'csv': 'to_csv'}


for values in selected_instruments.index:
    mask = df_meta[cols] == values
    logger.info(f"Samples: {mask.all(axis=1).sum()}")
    sample_ids = df_meta.loc[mask.all(axis=1)]
    display(sample_ids.sort_index())
    sample_ids = sample_ids.index
    # which categorical this might need to be a categorical Index as well?
    dataset = data.loc[sample_ids]
    # dataset.index = dataset.index.remove_unused_levels()

    display(dataset
            # .unstack(dataset.index.names[1:])
            .sort_index()
            )

    fname_dataset = vaep.io.get_fname_from_keys(values,
                                                file_ext=f".{FILE_EXT}")
    fname_dataset = (FOLDER_DATASETS /
                     fname_dataset.name.replace('Exactive_Series_slot_#', ''))
    files_out[fname_dataset.name] = fname_dataset
    logger.info(f'Dump dataset with N = {len(dataset)} to {fname_dataset}')
    _to_file_format = getattr(dataset, file_formats[FILE_EXT])
    _to_file_format(fname_dataset)

    # calculate support
    counts = dataset.count(axis=1).squeeze()
    ## to disk
    fname_support = vaep.io.get_fname_from_keys(values,
                                                folder='.',
                                                file_ext="")
    fname_support = (FOLDER_DATASETS /
                     (fname_support.stem + '_support.json').replace('Exactive_Series_slot_#', ''))
    files_out[fname_support.name] = fname_support
    logger.info(f"Dump support to: {fname_support.as_posix()}")
    
    counts.to_json(fname_support, indent=4)

    # very slow alternative, but 100% correct
    # M = dataset.index.droplevel(SAMPLE_ID).nunique()
    N, M = dataset.shape

    # plot support:
    fig, ax = plt.subplots()
    ax = (counts
          .sort_values()  # will raise an error with a DataFrame
          .reset_index(drop=True)
          .plot(rot=45,
                ax=ax,
                figsize=FIGSIZE,
                grid=True,
                xlabel='Count of samples ordered by number of features',
                title=f'Support of {len(counts):,d} samples features over {M} features ({", ".join(feat_name)})',
                ))
    vaep.plotting.add_prop_as_second_yaxis(ax, M)
    fig.tight_layout()
    fname_support = fname_support.with_suffix('.pdf')    
    files_out[fname_support.name] = fname_support
    vaep.plotting.savefig(fig, name=fname_support)

# %% [markdown]
# ## Last example dumped

# %%
dataset

# %%
# add json dump as target file for script for workflows
fname = FOLDER_DATASETS / 'selected_instruments.json'
files_out[fname.name] = fname
selected_instruments.to_json(fname, indent=4)
logger.info(f"Saved: {fname}")

# %%
files_out
