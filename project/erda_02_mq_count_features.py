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
# # Count peptides over all files

# %%
import os
import sys
import logging
from pathlib import Path
import random
import yaml
import json

import pandas as pd
import ipywidgets as widgets

### Logging setup ######
from vaep.logging import setup_nb_logger
setup_nb_logger()

### vaep imports ######
from vaep.io.mq import MaxQuantOutputDynamic
from vaep.io.data_objects import MqAllSummaries
from vaep.io.data_objects import PeptideCounter
import vaep.pandas

##################
##### CONFIG #####
##################
from config import FOLDER_MQ_TXT_DATA, FOLDER_PROCESSED

from config import FOLDER_DATA # project folder for storing the data
logging.info(f"Search Raw-Files on path: {FOLDER_MQ_TXT_DATA}")

# %% [markdown]
# Use samples previously loaded.

# %%
ELIGABLE_FILES_YAML = Path('config/eligable_files.yaml')
MAP_FOLDER_PATH = Path('config/file_paths')

with open(ELIGABLE_FILES_YAML) as f:
    files = set(yaml.safe_load(f)['files'])
    logging.info(f"Found a total of {len(files):,d} eligable files.")
with open(MAP_FOLDER_PATH) as f:
    folders_dict = yaml.safe_load(f)
    folders_dict = {folder: folders_dict[folder] for folder in files}  # only select folders selected

folders = [Path(folders_dict[folder]) for folder in files]
assert len(files) == len(folders_dict) == len(folders)

# %%
OVERWRITE = False

from config import FNAME_C_PEPTIDES, FNAME_C_EVIDENCE, FNAME_C_PG, FNAME_C_GENES

FNAME_C_PEPTIDES, FNAME_C_EVIDENCE, FNAME_C_PG, FNAME_C_GENES

# %% [markdown]
# ## Random example

# %%
import random
pd.set_option('max_columns', 60)
random_folder, random_path = random.sample(folders_dict.items(), 1)[0]
mq_output = MaxQuantOutputDynamic(random_path)
print(f"peptides.txt from {random_folder!s}")
mq_output.peptides

# %%
use_columns = mq_output.peptides.columns[33:45]
df = mq_output.peptides[use_columns].convert_dtypes() #.to_json('test.json')
df

# %%
df_json_string = df.to_json(orient='index', indent=4)
df_json_string[:1000]

# %%
df_csv = df.to_csv()
df_csv[:1000]

# %%
pd.read_json(df_json_string, orient='index')

# %%
mq_output.peptides.Intensity # as is in peptides.txt, comma seperated thousands

# %% [markdown]
# ## Count aggregated peptides

# %%
peptide_counter = PeptideCounter(FNAME_C_PEPTIDES, overwrite=OVERWRITE)
peptide_counter

# %%
if peptide_counter.loaded:
    print(peptide_counter.counter.most_common(10),
          len(peptide_counter.loaded),
          sep='\n')
else:
    print('New file created.')

# %% [markdown]
# - creates peptide intensity dumps for each MQ outputfolder per default `count_peptides` function (default processing function for `PeptideCounter`)

# %%
# %%time
# folders = [Path(folder_path) for folder_path in folders_dict.values()]
c = peptide_counter.sum_over_files(folders=folders)

# %%
c.most_common(10) # peptide_counter.counter.most_common(10)

# %%
# To share as python file
N = 1000
with open(FOLDER_PROCESSED / f'most_common_{10}_peptides.py', 'w') as f:
    f.write('import pandas as pd\n\n')
    
    #pprint.pformat list -> do this using standardlibrary
    # https://docs.python.org/3/library/pprint.html
    f.write(f"most_common = [\n  ")
    f.write(',\n  '.join(f"{str(t)}" for t in c.most_common(N)))
    f.write("\n]\n\n")
    
    #peptide_counter.loaded()
    
    f.write("pd.DataFrame.from_records(most_common, index='Sequence', columns=['Sequence', 'counts'])\n")

# %% [markdown] Collapsed="false"
# ## Peptides by charge
#
# - count peptides by charge state (which are aggregated in `peptides.txt`)

# %%
evidence_cols = vaep.pandas.get_columns_accessor(mq_output.evidence.reset_index())
evidence_cols # vaep.mq get this list

# %%
evidence = mq_output.evidence.set_index(evidence_cols.Charge, append=True)
evidence

# %% [markdown]
# Modifikationen könnten noch zum index hinzugefügt werden

# %%
evidence.Modifications.value_counts()

# %%
vaep.pandas.prop_unique_index(evidence)

# %% [markdown]
# Using the protein AA sequence and it's charge as identifiers, does not yield a unique index.
#
# First potential contaminants and peptides with zero intensity (or missing intensity) can be removed from the table.
#
# These are apparently peptides identified by an MS2 spectrum but which could not be quantified by a MS1 scans

# %%
mask =  evidence[evidence_cols.Intensity].isna()
evidence.loc[mask, evidence_cols.Type].value_counts()

# %%
evidence_cols = vaep.io.data_objects.evidence_cols
use_cols = [evidence_cols.mz, evidence_cols.Protein_group_IDs, evidence_cols.Intensity, evidence_cols.Score, evidence_cols.Potential_contaminant]

evidence_selected = vaep.io.data_objects.select_evidence(evidence[use_cols])
evidence_selected

# %%
evidence_selected = evidence_selected.sort_values(by=['Sequence', 'Charge', 'Score'], ascending=False)
evidence_selected

# %%
evidence_selected = vaep.pandas.select_max_by(evidence_selected.reset_index(), [evidence_cols.Sequence, evidence_cols.Charge], evidence_cols.Score)
evidence_selected

# %%
from collections import Counter
c = Counter()
c.update(evidence.index)
c.most_common(10)

# %%
example = evidence.loc[c.most_common(10)[0][0]]

vaep.pandas.show_columns_with_variation(example)

# %% [markdown]
# - `Type`: only `MULTI-MSMS` and `MULIT-SECPEP` are quantified (does this mean a matching MS1 spectrum?)

# %%
evidence[evidence_cols.Type].value_counts()

# %% [markdown]
# Some peptides can be assigned to different protein group IDs (razor peptides)
#  - option: discared non-unique peptides (and Protein group IDs can be already a combination of several isotopes)
#  - option: select on `Score` or `Intensity` (is there a relationship?)
#  - option: select based on `Number of isotopic peaks`

# %%
evidence[evidence_cols.Protein_group_IDs].value_counts()

# %% [markdown]
# ## Count peptides based on evidence files

# %%
evidence_counter = vaep.io.data_objects.EvidenceCounter(FNAME_C_EVIDENCE, overwrite=OVERWRITE)
c = evidence_counter.sum_over_files(folders=folders)

# %% [markdown]
# ## Protein Groups
#
# - protein groups between files
#     - aggregate by GENE ?
#     - 

# %%
mq_output.proteinGroups.describe(include='all')

# %%
pg_cols = vaep.pandas.get_columns_accessor(mq_output.proteinGroups.reset_index())
pg_cols

# %%
use_cols = [
# pg_cols.Protein_IDs,
 pg_cols.Majority_protein_IDs,
 pg_cols.Gene_names,
 pg_cols.Evidence_IDs,
 pg_cols.Q_value,
 pg_cols.Score,
 pg_cols.Only_identified_by_site,
 pg_cols.Reverse,
 pg_cols.Potential_contaminant,
 pg_cols.Intensity,
]

pd.options.display.max_rows = 100
pd.options.display.min_rows = 40
mask = mq_output.proteinGroups[[pg_cols.Only_identified_by_site, pg_cols.Reverse, pg_cols.Potential_contaminant]].notna().sum(axis=1) > 0
mq_output.proteinGroups.loc[mask, use_cols]

# %%
msg = "Omitting the data drops {0:.3f} % of the data."
print(msg.format(
mask.sum() / len(mask) * 100
))

# %%
selection = mq_output.proteinGroups.loc[~mask, use_cols]
gene_counts = selection[pg_cols.Gene_names].value_counts() # Gene Names not unique
msg = 'proportion of entries with non-unique genes: {:.3f}'
print(msg.format(gene_counts.loc[gene_counts > 1].sum() / gene_counts.sum()))
gene_counts.head(20)

# %%
mask = selection.Intensity > 0 
msg = "Proportion of non-zero Intensities: {:.3f} (zero_ count = {})"
print(msg.format(mask.sum() / len(mask), (~mask).sum()))
selection.loc[~mask]

# %%
selection = selection.loc[mask]

# %% [markdown]
# Some Proteins have no gene annotation
#   - P56181 -> mitochondrial
#
# In the online version of Uniprot these seems to be annotated (brief check). 
# So latest version probably has a gene annotation, so therefore these files are kept

# %%
gene_set = selection[pg_cols.Gene_names].str.split(';')

col_loc_gene_names = selection.columns.get_loc(pg_cols.Gene_names)
_ = selection.insert(col_loc_gene_names+1, 'Number of Genes', gene_set.apply(vaep.pandas.length))

mask = gene_set.isna()
selection.loc[mask]

# %%
cols = vaep.pandas.get_columns_accessor(selection)
gene_counts = vaep.pandas.counts_with_proportion(selection[cols.Number_of_Genes])
gene_counts

# %% [markdown]
# Most `proteinGroups` have single genes assigned to them. If one only looks at gene sets,
# one can increase uniquely identified `proteinGroups` further. 
#
# > Can `geneGroups` (sets of `Gene Names`) be used instead of `proteinGroups`?

# %%
gene_sets_counts = selection[cols.Gene_names].value_counts()
gene_sets_counts.value_counts()

# %% [markdown]
# Potential solutions:
# - summarize intensity per gene. One of the isoforms seems to have the major proportion of intensity assigned.
# - select maximum by score (higher scores seem to be related to higher intensity)

# %%
non_unique_genes = gene_sets_counts.loc[gene_sets_counts > 1].index

mask = selection[cols.Gene_names].isin(non_unique_genes)
selection.loc[mask].reset_index().set_index(cols.Gene_names).sort_index()

# %% [markdown]
# Protein Groups with Gene set with three and more genes:

# %%
selection.loc[selection[cols.Number_of_Genes] > 2]

# %%
logging.info(f"Selection shape before dropping duplicates by gene: {selection.shape}")
mask_no_gene = selection[pg_cols.Gene_names].isna()
selection_no_gene = selection.loc[mask_no_gene]
logging.info(f"Entries without any gene annotation: {len(selection_no_gene)}")
selection_no_gene

# %%
selection = vaep.pandas.select_max_by(df=selection.loc[~mask_no_gene].reset_index(), grouping_columns=[pg_cols.Gene_names], selection_column=pg_cols.Score)
logging.info(f"Selection shape after  dropping duplicates by gene: {selection.shape}")
selection = selection.set_index(pg_cols.Protein_IDs)
mask = selection[cols.Gene_names].isin(non_unique_genes)
selection.loc[mask]

# %%
selection = selection.append(selection_no_gene)

# %%
protein_groups_counter = vaep.io.data_objects.ProteinGroupsCounter(FNAME_C_PG, overwrite=OVERWRITE)
c = protein_groups_counter.sum_over_files(folders=folders)

# %%
vaep.pandas.counts_with_proportion(pd.Series(c)) # Most proteinGroups are unique

# %% [markdown]
# ### Count genes
# Genes sets could be used to identify common features.
#
# > The assignment of isoforms to one proteinGroup or another might be volatile.  
# > A single (unique) peptide could lead to different assignments.
# > Imputation on the evidence level could be a way to alleviate this problem
#
# - If genes set are not unique for a single run, one would have to decide which to take

# %%
gene_counter = vaep.io.data_objects.GeneCounter(FNAME_C_GENES, overwrite=OVERWRITE)

if not gene_counter.dumps:
    #empty dict, replace
    gene_counter.dumps = dict(protein_groups_counter.dumps) # prot proteinGroups files to GeneCounter
pg_dumps = list(gene_counter.dumps.values())

c_genes = gene_counter.sum_over_files(folders=pg_dumps)

c_genes = pd.Series(c_genes)
vaep.pandas.counts_with_proportion(c_genes) # Most proteinGroups are unique

# %% [markdown] Collapsed="false"
# ## Theoretial Peptides from used fasta-file
#
# > `01_explore_FASTA.ipynb` (formely `misc_FASTA_tryptic_digest.ipynb`)

# %% [markdown]
#
