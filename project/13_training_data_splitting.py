# ---
# jupyter:
#   jupytext:
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
# # Split up data into single datasets
#
# - create datasets per (set of) instruments for a specific experiments
# - drop some samples based on quality criteria

# %%
import logging
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
FIGSIZE=(15,10)

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
DUMP: str = erda_dumps.FN_PROTEIN_GROUPS
# DUMP: str = erda_dumps.FN_PEPTIDES
# DUMP: str = erda_dumps.FN_EVIDENCE
FOLDER_DATASETS: str = f'single_datasets/{DUMP.stem}'
N_MIN_INSTRUMENT = 300
META_DATA: str = 'data/files_selected_metadata.csv'
FILE_EXT = 'pkl'
SAMPLE_ID = 'Sample ID'
OUT_INFO = 'dataset_info' # saved as tex, xlsx and json

# %% [markdown]
# Make sure output folder exists

# %%
DUMP = Path(DUMP) # set parameter from cli or yaml to Path
FOLDER_DATASETS = defaults.FOLDER_DATA / FOLDER_DATASETS
FOLDER_DATASETS.mkdir(exist_ok=True, parents=True)
logger.info(f"Folder for datasets to be created: {FOLDER_DATASETS.absolute()}")

# %% [markdown]
# ## Dumps

# %% [markdown]
# - load dumps
# - load file to machine mappings

# %%
data = pd.read_pickle(DUMP)
data = data.squeeze() # In case it is a DataFrame, not a series (-> leads to MultiIndex)
name_data = data.name
logger.info(f"Number of rows (row = sample, feature, intensity): {len(data):,d}")
data

# %% [markdown]
# Make categorical index a normal string index (this lead to problems when selecting data using `loc` and grouping data as level of data could not easily be removed from MultiIndex)
#
# - see [blog](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a)

# %%
index_columns = data.index.names
data = data.reset_index()
print(data.memory_usage(deep=True))
cat_columns = data.columns[data.dtypes == 'category']
if not cat_columns.empty:
    data[cat_columns] = data[cat_columns].astype('object')
    print("non categorical: \n", data.memory_usage(deep=True))
    logger.warning("if time allows, this should be investigate -> use of loc with data which is not categorical")
data = data.set_index(index_columns)

# %% [markdown]
# ## Support per sample

# %%
idx_non_sample = list(data.index.names)
idx_non_sample.remove(SAMPLE_ID)
idx_non_sample

# %%
# M = data.index.droplevel(SAMPLE_ID).nunique() # very slow alternative, but 100% correct
M = vaep.io.filenames.read_M_features(DUMP.stem)
logger.info(f"Number of unqiue features: {M}")

# %%
counts = data.groupby(SAMPLE_ID).count().squeeze()
N = len(counts)
counts.to_json(FOLDER_DATASETS / 'support_all.json', indent=4)
ax = (counts
      .sort_values()  # will raise an error with a DataFrame
      .reset_index(drop=True)
      .plot(rot=45,
            figsize=FIGSIZE,
            grid=True,
            ylabel='number of features in sample',
            xlabel='Sample rank ordered by number of features', 
            title=f'Support of {N:,d} samples features over {M} features ({", ".join(idx_non_sample)})',
            ))
vaep.plotting.add_prop_as_second_yaxis(ax, M)
fig = ax.get_figure()
fig.tight_layout()
vaep.plotting.savefig(fig, name='support_all',
                      folder=FOLDER_DATASETS)


# %%
counts = data.groupby(idx_non_sample).count().squeeze()
counts.to_json(FOLDER_DATASETS / 'feat_completeness_all.json', indent=4)
ax = (counts
      .sort_values()  # will raise an error with a DataFrame
      .reset_index(drop=True)
      .plot(rot=45,
            figsize=FIGSIZE,
            grid=True,
            ylabel='number of samples per feature',
            xlabel='Feature rank ordered by number of samples', 
            title=f'Support of {len(counts):,d} features over {N} samples ({", ".join(idx_non_sample)})',
            ))
vaep.plotting.add_prop_as_second_yaxis(ax, N)
fig = ax.get_figure()
vaep.plotting.savefig(fig, name='feat_per_sample_all',
                      folder=FOLDER_DATASETS)

# %% [markdown]
# ## Filter for odd samples
#
# - fractionated samples
# - GPF - Gas phase fractionation # Faims? DIA? 
# - DIA
# - CV

# %%
# see 02_data_exploration_peptides

# %% [markdown]
# ## Meta Data
#
# - based on ThermoRawFileParser

# %%
# sample_ids = data.index.levels[0] # assume first index position is Sample ID?
sample_ids = data.index.get_level_values(SAMPLE_ID).unique() # more explict
sample_ids

# %%
df_meta = pd.read_csv(META_DATA, index_col=SAMPLE_ID)
date_col = 'Content Creation Date'
df_meta[date_col] = pd.to_datetime(df_meta[date_col])
df_meta = df_meta.loc[sample_ids]
df_meta

# %% [markdown]
# ### Available instruments

# %%
counts_instrument = df_meta.groupby(thermo_raw_files.cols_instrument)[date_col].agg(
    ['count', 'min', 'max']).sort_values(by=thermo_raw_files.cols_instrument[:2] + ['count'], ascending=False)
counts_instrument

# %%
len(counts_instrument)

# %%
selected_instruments = counts_instrument.query(f"count >= {N_MIN_INSTRUMENT}")
fname = FOLDER_DATASETS / 'dataset_info'
selected_instruments.to_latex(f"{fname}.tex")
selected_instruments.to_excel(f"{fname}.xlsx")
logger.info(f"Save Information to: {fname} (as json, tex)")
selected_instruments

# %% [markdown]
# ## Summary plot

# %%
reducer = umap.UMAP(random_state=42)
data = data.unstack(idx_non_sample)
data

# %%
embedding = reducer.fit_transform(data.fillna(data.median()))
embedding = pd.DataFrame(embedding, index=data.index, columns=['UMAP 1', 'UMAP 2'])
embedding = embedding.join(df_meta[["Content Creation Date", "instrument serial number"]])
embedding = embedding.join(counts_instrument['count'].reset_index(level=[0,1], drop=True), on='instrument serial number')
embedding

# %%
digits = int(np.ceil(np.log10(embedding["count"].max())))
digits

# %%
embedding["instrument with N"] = embedding[["instrument serial number", "count"]].apply(lambda s: f"{s[0]} (N={s[1]:{digits}d})", axis=1)
embedding["instrument with N"] = embedding["instrument with N"].str.replace('Exactive Series slot', 'Instrument')
embedding

# %% [markdown]
# define top five instruments

# %%
top_5 = counts_instrument["count"].nlargest(5)
top_5 = top_5.index.levels[-1]
embedding["instrument"] = embedding["instrument serial number"].apply(lambda x: x if x in top_5 else 'other')

# %%
embedding["Date (90 days intervals)"] = embedding["Content Creation Date"].dt.round("90D").astype(str)
to_plot = embedding.loc[embedding["instrument"] != 'other']
print(f"N samples in plot: {len(to_plot):,d}")
fig, ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(data=to_plot, x='UMAP 1', y='UMAP 2', style="instrument with N", hue="Date (90 days intervals)", ax=ax) #="Content Creation Date")
vaep.savefig(fig, name='umap_interval90days_top5_instruments', folder=FOLDER_DATASETS)

# %%
markers = ['o', 'x', 's', 'P', 'D', '.']
alpha = 0.6
fig, ax = plt.subplots(figsize=(12,8))
groups = list()

vaep.plotting.make_large_descriptors()
embedding["Content Creation Date"] = embedding["Content Creation Date"].dt.round("D")
embedding["mdate"] = embedding["Content Creation Date"].apply(matplotlib.dates.date2num)

to_plot = embedding.loc[embedding["instrument"] != 'other']

norm = matplotlib.colors.Normalize(embedding["mdate"].quantile(0.05), embedding["mdate"].quantile(0.95))
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
    
cbar = vaep.analyzers.analyzers.add_date_colorbar(ax.collections[0], ax=ax, fig=fig)
cbar.ax.set_ylabel("date of measurement", labelpad=-115, loc='center')
ax.legend(ax.collections, groups, title='instrument serial number')
ax.set_xlabel('UMAP 1') #, fontdict={'size': 16})
ax.set_ylabel('UMAP 2')
vaep.savefig(fig, name='umap_date_top5_instruments', folder=FOLDER_DATASETS)

# %%
fig,ax = plt.subplots(1,1, figsize=(8, 8))
vaep.plotting.make_large_descriptors()
to_plot = data.isna().sum(axis=0).reset_index(drop=True).to_frame('Feature prevalence')
to_plot = to_plot.join(data.isna().sum(axis=1).reset_index(drop=True).to_frame('Features per sample'))
to_plot = to_plot.join(counts_instrument.reset_index(drop=True)['count'].rename('Samples per instrument', axis='index'))
ax = to_plot.plot(kind='box', ax = ax, ylabel='number of observations')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
to_plot.to_csv(FOLDER_DATASETS/ 'summary_statistics_dump_data.csv')
vaep.savefig(fig, name='summary_statistics_dump',
                      folder=FOLDER_DATASETS)


# %%
data = data.stack(idx_non_sample)
data

# %% [markdown]
# ## Dump single experiments

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
    dataset = data.loc[sample_ids]  # which categorical this might need to be a categorical Index as well?
    dataset.index = dataset.index.remove_unused_levels()

    display(dataset
            .unstack(dataset.index.names[1:])
            .sort_index()
            )

    fname_dataset = vaep.io.get_fname_from_keys(values,
                                                folder=FOLDER_DATASETS,
                                                file_ext=f".{FILE_EXT}")

    logger.info(f'Dump dataset with N = {len(dataset)} to {fname_dataset}')
    _to_file_format = getattr(dataset, file_formats[FILE_EXT])
    _to_file_format(fname_dataset)

    fname_support = vaep.io.get_fname_from_keys(values,
                                                folder='.',
                                                file_ext="")
    fname_support = fname_support.stem + '_support'
    logger.info(f"Dump support to: {fname_support}")
    counts = dataset.groupby(SAMPLE_ID).count().squeeze()
    counts.to_json(FOLDER_DATASETS / f"{fname_support}.json", indent=4)

    M = dataset.index.droplevel(SAMPLE_ID).nunique()  # very slow alternative, but 100% correct

    # plot:
    fig, ax = plt.subplots()
    ax = (counts
          .sort_values()  # will raise an error with a DataFrame
          .reset_index(drop=True)
          .plot(rot=45,
                ax=ax,
                figsize=FIGSIZE,
                grid=True,
                xlabel='Count of samples ordered by number of features',
                title=f'Support of {len(counts):,d} samples features over {M} features ({", ".join(idx_non_sample)})',
                ))
    vaep.plotting.add_prop_as_second_yaxis(ax, M)
    fig.tight_layout()
    vaep.plotting.savefig(fig, name=fname_support,
                          folder=FOLDER_DATASETS)

# %% [markdown]
# ## Last example dumped

# %%
dataset

# %%
# add json dump as target file for script for workflows
selected_instruments.to_json(f"{fname}.json", indent=4)
logger.info(f"Saved: {fname}.json")
