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
# # RawFiles Database
#
# - overview of raw files, among others
#     - filesize
#     - duplicates of raw files
#     - search for substrings to find special cases (e.g. fractionated samples)
#
# **Outputs**
#
# Created data and figures
#
# ```bash
# 'data/all_raw_files_dump_duplicated.txt'
# 'data/all_raw_files_dump_unique.csv' # csv file
# 'Figures/raw_file_overview.pdf'
# ```
#
# **Inputs**
#
# ```bash
# 'data/all_raw_files_dump.txt'
# ```
#
# The ladder can be created using `find` on a server:
#
# ```bash
# find . -name '*.raw' -exec ls -l {} \; > all_raw_files_dump_2021_10_27.txt
# # alternative (changes the format)
# find . -name '*.raw' -ls > all_raw_files_dump_2021_10_27.txt
# ```
#
# which was executed in the 

# %%
from pathlib import Path, PurePosixPath
from collections import namedtuple
from functools import partial
import yaml

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging
from vaep.logging import setup_logger
from vaep.analyzers.analyzers import AnalyzePeptides
from vaep.io.data_objects import MqAllSummaries
from vaep.io.rawfiles import RawFileViewer, get_unique_stem, find_indices_containing_query, show_fractions
import config
from vaep.nb import Config
from vaep import utils

cfg = Config()

logger = logging.getLogger('vaep')
logger = setup_logger(logger, fname_base='00_2_hela_all_raw_files_ipynb')

# %% tags=["parameters"]
# FN_ALL_RAW_FILES = config.FOLDER_DATA / config.FN_ALL_RAW_FILES
FN_ALL_RAW_FILES: str = config.FOLDER_DATA / 'all_raw_files_dump_2021_10_29.txt'
FN_ALL_SUMMARIES: str = config.FN_ALL_SUMMARIES
FN_PEPTIDE_INTENSITIES = config.FOLDER_DATA / 'df_intensities_N07285_M01000' 

# %%
cfg.FN_ALL_RAW_FILES = FN_ALL_RAW_FILES
cfg.FN_ALL_SUMMARIES = FN_ALL_SUMMARIES
cfg.FN_PEPTIDE_INTENSITIES = FN_PEPTIDE_INTENSITIES

# %%
RawFile = namedtuple('RawFile', 'name path bytes')

data = []
with open(cfg.FN_ALL_RAW_FILES) as f:
    for line in f:
        line = line.split(maxsplit=8) # ignore white spaces in file names, example:
        #'-rw-r--r--. 1 501 501 282917566 Dec  3  2022 ./share_hela_raw/MNT_202220220921_EXLP1_Evo1_LiNi_ - Copy1.raw'
        path = Path(line[-1].strip())
        data.append(RawFile(path.stem, path, int(line[4])))

data = pd.DataFrame.from_records(
    data, columns=RawFile._fields, index=RawFile._fields[0])

data.sort_values(by='path', inplace=True)
data.head()

# %%
data['size_gb'] = data['bytes'] / 1024 ** 3
data

# %%
fname = 'processed/all_raw_file_sizes.csv'
data.to_csv(fname)

# %% [markdown]
# ## Finding duplicates
#
# - add a numeric index column to identify samples

# %%
data['num_index'] = pd.RangeIndex(stop=len(data))
mask_non_unique = data.reset_index().duplicated(subset=['name', 'bytes'])
mask_non_unique.index = data.index
idx_non_unique = data.loc[mask_non_unique].index.unique()
idx_non_unique # min number of files to remove


# %%
def check_for_duplicates(df):
    if df.index.is_unique:
        print('Only unique files in index.')
        return None
    else:
        non_unique = df.index.value_counts()
        non_unique = non_unique[non_unique > 1]
        # should this be browseable?
        print(f'Number of files with more than 2 duplicates: {(non_unique > 2).sum()}')
        return non_unique

non_unique = check_for_duplicates(df=data)
non_unique

# %% [markdown]
# Are there cases where only two files share the same name and have different file sizes:

# %%
data.loc[
    non_unique.index.difference(idx_non_unique) ]

# %% [markdown]
# For same sized groups, remove first the onces in the `MNT` folder:

# %%
data_in_MNT_to_remove = None
non_unique_remaining = None
if not data.index.is_unique:
    _data_to_remove = data.loc[idx_non_unique]
    data_in_MNT_to_remove = pd.DataFrame()
    non_unique_remaining = pd.DataFrame()
    for idx, g in _data_to_remove.groupby(level=0):
        mask = ['\\MNT' in str(x) for x in g.path]
        assert len(mask) != sum(mask) , f'All files in MNT subfolders: {idx}'
        data_in_MNT_to_remove = data_in_MNT_to_remove.append(g[mask])
        non_unique_remaining = non_unique_remaining.append(g[[x!=True for x in mask]])

    del _data_to_remove, mask, idx, g

assert len(data.loc[idx_non_unique]) == len(non_unique_remaining) + len(data_in_MNT_to_remove)
assert len(non_unique_remaining.loc[['\\MNT' in str(x) for x in non_unique_remaining.path]]) == 0, "There are files in MNT folder left"
data_in_MNT_to_remove

# %% [markdown]
# The main junk of duplicated files in in `MNT` subfolders

# %%
non_unique_remaining_counts = check_for_duplicates(non_unique_remaining)
non_unique_remaining.loc[non_unique_remaining_counts.index.unique()]

# %% [markdown]
# Files with the same name and the same size are considered the same.

# %%
mask_non_unique_remaining = non_unique_remaining.reset_index().duplicated(subset=['name', 'bytes'])
mask_non_unique_remaining.index = non_unique_remaining.index
data_to_remove = data_in_MNT_to_remove.append(
                    non_unique_remaining.loc[mask_non_unique_remaining]
)
data_to_remove

# %%
print(f"Save {data_to_remove['size_gb'].sum():1.0f} GB disk space by deleting {len(data_to_remove)} files.")

# %%
data_unique = data.reset_index().set_index('num_index').drop(data_to_remove.set_index('num_index').index).set_index('name')
data_unique

# %% [markdown]
# Make sure that every index to remove is still present in `data_unique` which is data to keep

# %%
data_unique.loc[data_to_remove.index.unique()]

# %%
assert len(data_unique) + len(data_to_remove)  == len(data)

# %% [markdown]
# Show files which are duplicated, but have different sizes:

# %%
# two files have the same name, but different sizes
data_unique.loc[data_unique.index.duplicated(False)] if not data_unique.index.is_unique else None

# %% [markdown]
# Save unique files

# %%
cfg.FN_ALL_RAW_FILES_UNIQUE = utils.append_to_filepath(cfg.FN_ALL_RAW_FILES, config.build_df_fname(data_unique, 'unique'), new_suffix='csv')
data_unique.to_csv(cfg.FN_ALL_RAW_FILES_UNIQUE)

# %% [markdown]
# Export file paths to file to remove them, e.g using `rm $(<filenames.txt))` following [this description](https://stackoverflow.com/a/18618543/9684872).
#
# ```bash
# # remove empty lines
# cat all_raw_files_dump_duplicated.txt | grep .raw > all_raw_files_dump_duplicated_cleaned.txt
# ls `cat all_raw_files_dump_duplicated_cleaned`
# rm -i `cat all_raw_files_dump_duplicated_cleaned`
# rm -i $(<all_raw_files_dump_duplicated_cleaned.txt)
# ```

# %%
cfg.FN_ALL_RAW_FILES_DUPLICATED = utils.append_to_filepath(cfg.FN_ALL_RAW_FILES, 'duplicated')

with open(cfg.FN_ALL_RAW_FILES_DUPLICATED, 'w') as f:
    for _path in data_to_remove['path']:
        _path = PurePosixPath(_path)
        f.write(f'{_path}\r\n')

# %%
fig, axes = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [
                         5, 1], "wspace": 0.3}, figsize=(16, 8))
data_unique['size_gb'].plot.hist(bins=30, ax=axes[0])
data_unique['size_gb'].plot(kind='box', ax=axes[1])


cfg.raw_file_overview = config.FIGUREFOLDER / 'raw_file_overview.pdf'

fig.savefig(cfg.raw_file_overview)

# %%
data_unique.describe(np.linspace(0.1, 0.9, 9))

# %% [markdown]
# ## Find fractionated samples for raw files
#
# - franctionated samples need to be processed together

# %%
viewer = RawFileViewer(data_unique, outputfolder=config.FOLDER_DATA)
_ = viewer.view()
display(_)

# %% [markdown]
# ### Query: fractionated samples
#
# hard coded query to output fractionated samples

# %%
file_names = data_unique.index

find_indices_containing_query = partial(find_indices_containing_query, X=data_unique)

# %%
q = '[Ff]rac'  # query field
df_selected = find_indices_containing_query(q)

# %%
frac_unique = get_unique_stem(q, df_selected.index)

# %%
# samples where current approach of spliting based on frac does not work.
# frac denotes here the total number of fractions (3, 6, 8, 12, 24, 46)

frac_special_cases = [
    # continue with samples below 2019 (select in DropDown below)
    '20180508_QE3_nLC5_DBJ_DIAprot_HELA_500ng_GPF',
    '20180528_QE5_Evo2_DBJ_DIAprot_HeLa_500ng',
    '20190108_QE7_Evo1_DBJ_SA_LFQpho_HELA_PACs_200ug', # s mssing in LFQphos
    '20190108_QE7_Evo1_DBJ_SA_LFQphos_HELA_PAC_200ug',
    '20190108_QE7_Evo1_DBJ_SA_LFQphos_HELA_PAC_300ug',
    '20190108_QE7_Evo1_DBJ_SA_LFQphos_HELA_PAC_400ug',
    '20190212_QE5_Evo1_DBJ_LFQprot',
    '20190314_QE3_DBJ_Evo2_LFQphos_Hela_200ug_StageTip',
    '20190314_QE3_DBJ_Evo2_LFQphos_Hela_380ug_StageTip', # first t missing in StagetTip
    '20190314_QE3_DBJ_Evo2_LFQphos_Hela_380ug_StagetTip',
    '20190402_QE3_Evo1_DBJ_DIAprot_HELA',
    '20190402_QE3_Evo1_DBJ_LFQprot_HELA',
    '20190430_QE3_Evo2_DBJ_HELA_14cmCol_60degrees_5min',
    '20190430_QE3_Evo2_DBJ_LFQprot_HELA-14cmCol_44min',
    '20190507_QE5_Evo1_DBJ_LFQprot_Subcell_HeLa_Ctrl',
    '20190507_QE5_Evo1_DBJ_LFQprot_Subcell_library_HeLa_Ctrl_Ani_Mix',
    '20190622_EXP1_Evo1_AMV_SubCell-library-HeLa_21min-30000',
    '20190628_EXP1_Evo1_AMV_SubCell-library-HeLa_21min-30000',   
]

# exclude keys and handle separately. Remaining keys can be used directly to create list of inputs.
frac_unique = sorted(list(set(frac_unique) - set(frac_special_cases)))

# %%
w_data = widgets.Dropdown(options=frac_unique, index=0)
show_fractions_frac = partial(show_fractions, df=df_selected)
out_sel = widgets.interactive_output(show_fractions_frac, {'stub': w_data})
widgets.VBox([w_data, out_sel]) # repr of class
#stub, export

# %% [markdown]
# - `frac12` indicates 12 splits. If there are more, some of them were re-measured, e.g. `0190920_QE3_nLC3_MJ_pSILAC_HeLa_48h_Frac01_Rep3_20190924081042`
#

# %% [markdown]
# ## For quantified samples
# - show scatter plot between sample size and number of quantified peptides

# %%
common_peptides = AnalyzePeptides.from_csv(cfg.FN_PEPTIDE_INTENSITIES)
common_peptides = common_peptides.df.index

# %%
mq_summaries = MqAllSummaries(cfg.FN_ALL_SUMMARIES)

# %% [markdown]
# Only keep one copy of files with the same name

# %%
data_unique.loc[data_unique.index.duplicated(False)] if not data_unique.index.is_unique else None

# %%
data_unique_index = data_unique.index.duplicated()
data_unique_index = data_unique.loc[~data_unique_index]

# %%
_idx_missing = mq_summaries.df.index.difference(data_unique_index.index)
assert not len(_idx_missing), f"There are missing files processed in the list of raw files: {_idx_missing}"

# %% [markdown]
# > They can be duplicated files with the same file size. Not the case for now

# %%
idx_shared = mq_summaries.df.index.intersection(data_unique.index)

_file_sizes = data_unique.loc[idx_shared, 'size_gb']
_file_sizes.loc[_file_sizes.index.duplicated(False)]

# %%
_file_sizes = _file_sizes.loc[~_file_sizes.index.duplicated(keep='last')]
mq_summaries.df.loc[idx_shared, 'file size in GB'] = _file_sizes
cols = ['Peptide Sequences Identified', 'file size in GB']
mq_summaries.df[cols]

# %%
mq_summaries.df[cols].describe(np.linspace(0.05, 0.95, 10))

# %%
fig, axes = plt.subplots(ncols=3, gridspec_kw={"width_ratios": [
                         5, 1, 1], "wspace": 0.3}, figsize=(20, 8))

ax = axes[0]
ax = mq_summaries.df.plot.scatter(x=cols[0], y=cols[1], ax=ax)
ax.axvline(x=15000)

ax = axes[1]
ax = mq_summaries.df[cols[0]].plot(kind='box', ax=ax)


ax = axes[2]
ax = mq_summaries.df[cols[1]].plot(kind='box', ax=ax)

# %% [markdown]
# For some files with a large number of identified peptides, the file size information seems to be missing.

# %%
cfg.figure_1 = config.FIGUREFOLDER / 'figure_1.pdf'

fig.savefig(cfg.figure_1)

# %%
threshold = 15_000
mask = mq_summaries.df[cols[0]] > threshold
print(
    f"for threshold of {threshold:,d} quantified peptides:\n"
    f"Total number of files is {mask.sum()}\n"
    "Minimum file-size is {:.3f} GB.\n".format(
        mq_summaries.df.loc[mask, cols[1]].min())
)

# %% [markdown]
# ## Meta data for all samples

# %% [markdown]
# ### From raw file reading

# %%
files_to_parse = data_unique.loc[idx_shared, 'path'].apply(lambda path: str(PurePosixPath(path)).strip())
files_to_parse = dict(files=files_to_parse.to_list())
cfg.remote_files = config.FOLDER_DATA / 'remote_files.yaml'
with open(cfg.remote_files, 'w') as f:
    yaml.dump(files_to_parse, f)
print(f"Saved list of files to: {cfg.remote_files}")

# %% [markdown]
# ### From file name

# %%
analysis = AnalyzePeptides.from_csv(cfg.FN_ALL_RAW_FILES_UNIQUE,index_col='name') # ToDo: Add numbers to file names
analysis.df

# %%
analysis.add_metadata(add_prop_not_na=False)

# %% [markdown]
# Metadata has fewer cases due to duplicates with differnt file sizes ( see above)

# %%
analysis.df.loc[analysis.df.index.duplicated(False)] # keep the larger one

# %% [markdown]
# ## cfg

# %%
vars(cfg) # return a dict which is rendered differently in ipython

# %%
