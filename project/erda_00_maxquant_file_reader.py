# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.8.12 ('vaep')
#     language: python
#     name: python3
# ---

# %% [markdown] Collapsed="false"
# # MaxQuant (MQ) Output-Files
#
# Compare a single experiment
#
# Files compared:
# 1. `Summary.txt`
# 2. `mqpar.xml`
# 3. `peptides.txt`
# 4. `proteins.txt`
#
# There is are many files more, where several files seem to be available in several times in different formats.

# %%
import os
import sys
import logging
from pathlib import Path
import random
from tqdm.notebook import tqdm

import pandas as pd
import ipywidgets as widgets

from vaep.io import PathsList
from vaep.io.mq import MaxQuantOutputDynamic
from vaep.io.mq import ExtractFromPeptidesTxt
import vaep.io.mq as mq


from src.file_utils import load_summary, load_mqpar_xml
from vaep.logging import setup_logger_w_file

##################
##### CONFIG #####
##################
from config import FOLDER_MQ_TXT_DATA, FOLDER_PROCESSED
from config import FOLDER_KEY  # defines how filenames are parsed for use as indices

from config import FOLDER_DATA # project folder for storing the data
print(f"Search Raw-Files on path: {FOLDER_MQ_TXT_DATA}")

##################
### Logging ######
##################

#Delete Jupyter notebook root logger handler
root_logger = logging.getLogger()
root_logger.handlers = []

logger = logging.getLogger('vaep')
logger = setup_logger_w_file(logger, fname_base='log_00_maxquant_file_reader')

logger.info('Start with handlers: \n' + "\n".join(f"- {repr(log_)}" for log_ in logger.handlers))

# %%
folders = [folder for folder in  Path(FOLDER_MQ_TXT_DATA).iterdir() if folder.is_dir()]

# %%
folders_dict = {folder.name: folder for folder in sorted(folders) }
assert len(folders_dict) == len(folders), "Non unique file names"

# %% Collapsed="false"
# w_file = widgets.Dropdown(options=[folder for folder in folders], description='View files')
w_file = widgets.Dropdown(options=folders_dict, description='View files')
w_file

# %%
mq_output = MaxQuantOutputDynamic(w_file.value)
mq_output

# %% [markdown]
# Results will be saved in a subfolder under `vaep/project/data` using the name of the specified input-folder per default. Change to your liking:

# %% [markdown]
# > Go to the block you are interested in!

# %% [markdown] Collapsed="false"
# ## MQ Summary files

# %%
mq_output.summary.iloc[0].to_dict()

# %% [markdown] Collapsed="false"
# ### File Handler
#
# - dictionary of run name to run output folder
# - find class with expected output folders

# %% Collapsed="false"
# # load_summary??

# %% [markdown] Collapsed="false"
# ### Summaries
#
# - aggregated in `vaep/project/erda_01_mq_aggregate_summaries.ipynb` 
#     - file selection based on summaries for further analysis thereafter

# %%
# paths_summaries = [str(folder / 'summary.txt') for folder in folders_dict.values()]

# %% Collapsed="false"
# # if paths_summaries.files:
# if folders_dict:
# #     df, names, failed = process_files(handler_fct=load_summary, filepaths=paths_summaries.files, key=FOLDER_KEY, relative_to='paths_summaries.folder')
#     df, names, failed = process_files(handler_fct=load_summary, filepaths=paths_summaries, key=FOLDER_KEY, relative_to=None)
#     df.columns = names
#     print(f"Number of failed reads: {len(failed)}")
#     display(df)

# %%
# # if paths_summaries.files:
# if paths_summaries:
#     df.to_csv(os.path.join(FOLDER_PROCESSED, 'all_summary_txt.csv'))
#     df.to_pickle(os.path.join(FOLDER_PROCESSED, 'all_summary_txt.pkl'))

# %% [markdown]
# - SIL - MS2 based on precursor which was a set of peaks
# - PEAK - MS2 scan based on a single peak on precursor spectrum
# - ISO - isotopic pattern detection
#

# %%
# # if paths_summaries.files:
# if paths_summaries:
#     MS_spectra = df.loc[['MS', 'MS/MS Identified']].T.astype('int64')
#     mask  = MS_spectra['MS/MS Identified'] > 0
#     display(MS_spectra.loc[mask].describe())
#     MS_spectra.to_csv(os.path.join(FOLDER_PROCESSED, 'overview_stats.csv'))

# %% [markdown] Collapsed="false"
# ## MaxQuant Parameter File
#
# - partly in a separate subfolder
# - mainly in run folders
# - rebase on folders_dictionary (check for `.xml` files in all folders)

# %%
mqpar_files = (Path(FOLDER_DATA) / 'mqpar_files')

mqpar_files  = [file for file in mqpar_files.iterdir() if file.suffix == '.xml']
len(mqpar_files) # nested search needed

# %% Collapsed="false"
w_file = widgets.Dropdown(options=mqpar_files, description='Select a file')
w_file

# %% [markdown] Collapsed="false"
# ### Parameter Files

# %% Collapsed="false"
fname_mqpar_xml = os.path.join(FOLDER_PROCESSED, 'peptide_intensities.{}')

d_mqpar = dict()
for file in tqdm(mqpar_files):
    d_mqpar[file.stem] = load_mqpar_xml(file)['MaxQuantParams']
    
df_mqpar = pd.DataFrame(d_mqpar.values() , index=d_mqpar.keys()).convert_dtypes()
df_mqpar

# %% [markdown]
# The number of threads used might differ

# %%
df_mqpar['numThreads'].value_counts()

# %% [markdown]
# The parameter files would need further parsing, which is skipped for now:
#  - `OrderedDict` would need to be flattend
#  - in the example below, it is not easy to see how entries should be easily combined
#     (list of `OrderedDict`s where only the `fastaFilePath` is different)

# %%
df_mqpar.iloc[0].loc['fastaFiles']

# %% [markdown]
# in order to see if there are different setting based on the string columns, drop duplicates 
#
# - only one should remain

# %%
df_mqpar.select_dtypes('string').drop('numThreads', axis=1).drop_duplicates()

# %% [markdown] Collapsed="false"
# ## Peptides
#
# - peptides combined (combining different charged states): `peptides`
# - single peptides (with differing charges): `evidence`

# %% Collapsed="false"
pd.set_option('max_columns', 60)

# mq_output = MaxQuantOutputDynamic(
#     folder=folders[random.randint(0, len(paths_peptides.files)-1)])
mq_output.peptides

# %%
mq_output.evidence

# %%
mq_output.peptides.Intensity # as is in peptides.txt, comma seperated thousands

# %% [markdown] Collapsed="false"
# ### Create peptide intensity dumps for each MQ outputfolder
#
# - idea was: dump peptides found for each (unique) gene
#     - creates a `json` file for each gene with the gene contained
#
# - decision: discard
#     - rather dump peptide information per sample. Mapping of peptides to gene can be done
#       using the fasta file on the pytorch level.

# %%
# folders[:10]

# %% [markdown]
# Check if the output folder contains already parsed files

# %%
# import json

# with open(src.config.FN_FASTA_DB) as f:
#     data_fasta = json.load(f)
# print(f'Number of proteins in fasta file DB: {len(data_fasta)}')

# %%
# # %%time
# FOLDER_PEP_PER_GENE = Path(FOLDER_PROCESSED) / 'agg_peptides_per_gene'
# FOLDER_PEP_PER_GENE.mkdir(parents=True, exist_ok=True)
# set_previously_loaded =  {folder.name for folder in FOLDER_PEP_PER_GENE.iterdir()}

# FORCE = True

# for folder in folders:
#     if folder.name in set_previously_loaded and not FORCE and (folder / '0_completness_all_genes.json').exists():
#         pass
#     else:
#         logger.info(f'\n\nProcess: {folder.name}')
#         mq_output = MaxQuantOutputDynamic(folder)
#         peptide_extractor = ExtractFromPeptidesTxt(
#             out_folder=FOLDER_PEP_PER_GENE, mq_output_object=mq_output, fasta_db=data_fasta)
#         completeness_per_gene = peptide_extractor()


# %% [markdown] Collapsed="false"
# ## Theoretial Peptides from used fasta-file
#
# > `01_explore_FASTA.ipynb` (formely `misc_FASTA_tryptic_digest.ipynb`)
#
# - check if peptides are part of theoretical peptides

# %% Collapsed="false"
