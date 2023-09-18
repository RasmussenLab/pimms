# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3.8.13 ('vaep')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore MaxQuant (MQ) output files of single runs
#
# The `erda_03_training_data.ipynb` notebook does extract information to be used as training data.
# File specific one could also use the retention time analysis to identify _valid_ co-occurring peptides to be use during training.
# Potentially this preprocessing step can be used at inference time.
#
# This notebook contains some relevant analysis for a specific `txt` output-folder in the current project
#
# ##### Analysis overview
#
# > Report for example data
#
# - relation between `peptides.txt` and `evidence.txt`

# %%
import json
import logging
import random

import ipywidgets as widgets

from numpy.testing import assert_almost_equal
from numpy import random
import pandas as pd
# pd.options.display.float_format = '{:,.1f}'.format

import vaep.pandas
from vaep.pandas import length
from vaep.io.mq import FASTA_KEYS, MaxQuantOutput
from vaep.io import search_subfolders
import vaep.io.mq as mq
from vaep.io.mq import mq_col

##### CONFIG #####
import config
from config import FOLDER_MQ_TXT_DATA as FOLDER_RAW_DATA
from config import FIGUREFOLDER


from vaep.logging import setup_nb_logger
logger = setup_nb_logger()


print(f"Search Raw-Files on path: {FOLDER_RAW_DATA}")

# %%
folders = search_subfolders(path=FOLDER_RAW_DATA, depth=1, exclude_root=True)
w_folder = widgets.Dropdown(options=folders, description='Select a folder')
w_folder

# %%
mq_output = MaxQuantOutput(folder=w_folder.value)

# %% [markdown]
# ## Some important columns
#
# Grouped by a namedtuple allowing attribute access

# %%
mq_col

# %% [markdown]
# ## `peptides.txt`
#
# > For reference on final "result"

# %%
pd.options.display.max_columns = len(mq_output.peptides.columns)
mq_output.peptides

# %% [markdown]
# `peptides.txt` contains aggregated peptides

# %%
intensities = mq_output.peptides.Intensity
intensities

# %% [markdown]
# Not all peptides are associated with a Protein or Gene by MQ, although
# there is evidence for the peptide. This is due to potential
# `CON_`taminants in the medium which is encouded by default by MQ.

# %%
mq_output.peptides[FASTA_KEYS].isna().sum()

# %% [markdown]
# ## `evidence.txt`
#
# contains
# - retention time for peptides
# - has repeated measures of the same sequences, which are all aggregated in `peptides.txt`
#

# %%
pd.options.display.max_columns = len(mq_output.evidence.columns)
mq_output.evidence

# %%
mq_output.evidence.Charge.value_counts().sort_index()

# %%
mask = mq_output.evidence[mq_col.RETENTION_TIME] != mq_output.evidence[mq_col.CALIBRATED_RETENTION_TIME]
print("Number of non-matching retention times between calibrated and non-calibrated column:", mask.sum())

# try:
#     assert mask.sum() == 0, "More than one replica?"
# except AssertionError as e:
#     logger.warning(e)
assert mask.sum() == 0, "More than one replica?"

# %% [markdown]
# Using only one quality control sample, leaves the initial retention time as is.

# %%
rt = mq_output.evidence[mq_col.CALIBRATED_RETENTION_TIME]

# %%
pep_measured_freq_in_evidence = rt.index.value_counts()
pep_measured_freq_in_evidence.iloc[:10]  # top10 repeatedly measured peptides

# %%
max_observed_pep_evidence = pep_measured_freq_in_evidence.index[0]
mq_output.evidence.loc[
    max_observed_pep_evidence,
    :
]

# %% [markdown]
# The retention time index is non-unique.

# %%
print('The retention time index is unique: {}'.format(rt.index.is_unique))

# %% [markdown]
# Peptides observed more than once at different times.

# %%
mask_duplicated = rt.index.duplicated(keep=False)
rt_duplicates = rt.loc[mask_duplicated]
rt_duplicates

# %%
mq_output.evidence.loc[mask_duplicated, [
    'Charge', 'Calibrated retention time', 'Intensity']]

# %% [markdown]
# Calculate median intensity and calculate standard deviation

# %%
_agg_functions = ['median', 'std']

rt_summary = rt.groupby(level=0).agg(_agg_functions)
rt_summary

# %% [markdown]
# Let's see several quartiles for both median and standard deviation (the
# columns are independent from each other) for the retention time

# %%
rt_summary.describe(percentiles=[0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99])

# %%
rt_summary['median']

# %% [markdown]
# A large standard-deviation indicates that the intensity values originate from time points (in min) widely spread.

# %% [markdown]
# ### Peptides observed several times a different points of experimental run

# %%
mask = rt_summary['std'] > 30.0
mask_indices = mask[mask].index
rt.loc[mask_indices]

# %% [markdown]
# Peptides with differen RT have different charges.

# %%
mq_output.evidence.loc[mask_indices]

# %% [markdown]
# Model evaluation possibility: Discard samples with several measurements
# from an experiment and predict value. See which intensity measurement
# corresponds more closely.

# %%
_peptide = random.choice(mask_indices)

# %%
f'evidence_{_peptide}_{w_folder.value.stem}'

# %%

peptide_view = mq_output.evidence.loc[_peptide]
peptide_view = (peptide_view[
    vaep.pandas.get_unique_non_unique_columns(peptide_view).non_unique]
    .dropna(axis=1, how='all')
    .set_index('Charge', append=True))
peptide_view

# %%
fname = w_folder.value.parent / f'evidence_{_peptide}_{w_folder.value.stem}.xlsx'
peptide_view.to_excel(fname)
fname

# %% [markdown]
# `Type` column indicates if peptide is based on one or more MS-MS spectra.

# %%
mq_output.peptides.loc[_peptide].to_frame().T

# %% [markdown]
# ## Differences in intensities b/w peptides.txt and evidence.txt
#
#
# The intensity reported in `peptides.txt` corresponds to roughly to the
# sum of the intensities found in different scans:

# %%
col_intensity = mq_col.INTENSITY
try:

    assert_almost_equal(
        _pep_int_evidence := mq_output.evidence.loc[_peptide, col_intensity].sum(),
        _pep_int_peptides := mq_output.peptides.loc[_peptide, col_intensity],
        err_msg='Mismatch between evidence.txt summed peptide intensities to reported peptide intensities in peptides.txt')
except AssertionError as e:
    logging.error(
        f"{e}\n Difference: {_pep_int_evidence - _pep_int_peptides:,.2f}")

# %%
mq_output.evidence.loc[_peptide, col_intensity]

# %%
mq_output.peptides.loc[_peptide, col_intensity]

# %% [markdown]
# Make this comparison for all peptides

# %%
_pep_int_evidence = mq_output.evidence.groupby(
    level=0).agg({col_intensity: [sum, len]})
_pep_int_evidence.columns = [col_intensity, 'count']
_pep_int_evidence

# %%
_diff = _pep_int_evidence[col_intensity] - \
    mq_output.peptides[col_intensity].astype(float)
mask_diff = _diff != 0.0
_pep_int_evidence.loc[mask_diff].describe()

# %%
_diff.loc[mask_diff]

# %%
_diff[mask_diff].describe()

# %% [markdown]
# Several smaller and larger differences in an intensity range way below the detection limit arise for some sequences.

# %% [markdown]
# ### Ideas on source of difference
#  - Are all peptides (sequences) which are based on single observations in `evidence.txt` represented as is in `peptides.txt`?
#  - how many peptides with more than one PTM have non-zero differences between the sum of intensity values in `evidence.txt` and the respective value in `peptides.txt`?
#  - maybe some peptides are filtered based on assignment as contaminent (`CON__`)?

# %%
# ToDo see above

# %%
_diff_indices = _diff[mask_diff].index
# some pep-seq in peptides.txt not in evidence.txt
_diff_indices = _diff_indices.intersection(mq_output.evidence.index.unique())

# %%
sample_index = random.choice(_diff_indices)

# %%
mq_output.evidence.loc[sample_index]

# %%
mq_output.peptides.loc[sample_index].to_frame().T

# %% [markdown]
# ### Modifications

# %%
mq_output.evidence.Modifications.value_counts()

# %% [markdown]
# ### Potential contaminant peptides

# %% [markdown]
# The `CON__` entries are possible contaminations resulting from sample preparation using a e.g. a serum:
#
# ```python
# data_fasta['ENSEMBL:ENSBTAP00000024146']
# data_fasta['P12763'] # bovine serum protein -> present in cell cultures and in list of default contaminant in MQ
# data_fasta['P00735'] # also bovin serum protein
# ```

# %%
mask = mq_output.peptides['Potential contaminant'].notna()
mq_output.peptides.loc[mask]

# %% [markdown]
# ### Aggregate identifiers in evidence.txt
#
# - `Proteins`: All proteins that contain peptide sequence

# %%
fasta_keys = ["Proteins", "Leading proteins",
              "Leading razor protein", "Gene names"]
mq_output.evidence[fasta_keys]

# %% [markdown]
# The protein assignment information is not entirely unique for each group of peptides.

# %% [markdown]
# ## align intensities and retention time (RT) for peptides
#
# - intensities are values reported in `peptides.txt`
# - some (few) peptides in `peptides.txt` are not in `evidence.txt`, but then probably zero

# %%
intensities.index

# %%
seq_w_summed_intensities = intensities.to_frame().merge(
    rt_summary, left_index=True, right_index=True, how='left')

# %%
seq_w_summed_intensities

# %%
mask = ~mq_output.evidence.reset_index(
)[["Sequence", "Proteins", "Gene names"]].duplicated()
mask.index = mq_output.evidence.index

# %%
diff_ = seq_w_summed_intensities.index.unique().difference(mask.index.unique())
diff_.to_list()

# %%
# mq_output.msms.set_index('Sequence').loc['GIPNMLLSEEETES']

# %%
# There is no evidence, but then it is reported in peptides?!
# Is this the case for more than one MQ-RUN (last or first not written to file?)
try:
    if len(diff_) > 0:
        mq_output.evidence.loc[diff_]
except KeyError as e:
    logging.error(e)

# %%
mq_output.peptides.loc[diff_]

# %% [markdown]
# ### Option: Peptide scan with highest score for repeatedly measured peptides
#
# - only select one of repeated peptide scans, namely the one with the highest score
# - discards information, no summation of peptide intensities
# - yields unique retention time per peptide, by discarding additional information

# %%
COL_SCORE = 'Score'
mq_output.evidence.groupby(level=0)[COL_SCORE].max()

# %%
mask_max_per_seq = mq_output.evidence.groupby(
    level=0)[COL_SCORE].transform("max").eq(mq_output.evidence[COL_SCORE])
mask_intensity_not_na = mq_output.evidence.Intensity.notna()
mask = mask_max_per_seq & mask_intensity_not_na

# %% [markdown]
# This leads to a non-unique mapping, as some scores are exactly the same for two peptides.

# %%
mask_duplicates = mq_output.evidence.loc[mask].sort_values(
    mq_col.INTENSITY).index.duplicated()
sequences_duplicated = mq_output.evidence.loc[mask].index[mask_duplicates]
mq_output.evidence.loc[mask].loc[sequences_duplicated, [
    COL_SCORE, mq_col.INTENSITY, mq_col.RETENTION_TIME]]  # .groupby(level=0).agg({mq_col.INTENSITY : max})

# %%
mask = mq_output.evidence.reset_index().sort_values(
    by=["Sequence", "Score", mq_col.INTENSITY]).duplicated(subset=["Sequence", "Score"], keep='last')
_sequences = mq_output.evidence.index[mask]
mq_output.evidence.loc[_sequences, [
    "Score", "Retention time", mq_col.INTENSITY, "Proteins"]]

# %% [markdown]
# - random, non missing intensity?

# %%
aggregators = ["Sequence", "Score", mq_col.INTENSITY]
mask_intensity_not_na = mq_output.evidence.Intensity.notna()
seq_max_score_max_intensity = mq_output.evidence.loc[mask_intensity_not_na].reset_index(
)[aggregators + ["Proteins", "Gene names"]].sort_values(by=aggregators).set_index("Sequence").groupby(level=0).last()
seq_max_score_max_intensity

# %%
# drop NA intensities first.
assert seq_max_score_max_intensity.Intensity.isna().sum() == 0

# %% [markdown]
# Certain peptides have no Protein or gene assigned.

# %%
seq_max_score_max_intensity.isna().sum()

# %%
mask_seq_selected_not_assigned = seq_max_score_max_intensity.Proteins.isna(
) | seq_max_score_max_intensity["Gene names"].isna()
seq_max_score_max_intensity.loc[mask_seq_selected_not_assigned]

# %% [markdown]
# These might be a candiate for evaluating predictions, as the information is measured, but unknown.
# If they cannot be assigned, the closest fit on different genes with
# model predictions could be a criterion for selection

# %% [markdown]
# ## Create dumps of intensities in `peptides.txt`

# %%
# mq_output.evidence.loc["AAAGGGGGGAAAAGR"]

# %%
# ToDo: dump this?
mq_output.dump_intensity(folder='data/peptides_txt_intensities/')

# %% [markdown]
# ## Create dumps per gene

# %% [markdown]
# Some hundred peptides map to more than two genes

# %%
seq_max_score_max_intensity[mq_col.GENE_NAMES].str.split(";"
                                                         ).apply(lambda x: length(x)
                                                                 ).value_counts(
).sort_index()

# %% [markdown]
# Mostly unique genes associated with a peptide.

# %% [markdown]
# ### Select sensible training data per gene
# - sequence coverage information?
# - minimal number or minimal sequence coverage, otherwise discared
# - multiple genes:
#     - select first and add reference in others
#     - split and dump repeatedly
#
# Load fasta-file information

# %%
with open(config.FN_FASTA_DB) as f:
    data_fasta = json.load(f)
print(f'Number of proteins in fasta file DB: {len(data_fasta)}')

# %%
# schema validation? Load class with schema?
# -> Fasta-File creation should save schema with it

# %% [markdown]
# ### Fasta Entries considered as contaminants by MQ

# %%
mask_potential_contaminant = mq_output.peptides['Potential contaminant'] == '+'
contaminants = mq_output.peptides.loc[mask_potential_contaminant, [mq_col.PROTEINS, mq_col.LEADING_RAZOR_PROTEIN]]
contaminants.head()

# %%
unique_cont = contaminants[mq_col.PROTEINS].str.split(';').to_list()
set_all = set().union(*unique_cont)
set_cont = {x.split('CON__')[-1] for x in set_all if 'CON__' in x}
set_proteins_to_remove = set_all.intersection(set_cont)
set_proteins_to_remove

# %% [markdown]
# List of proteins which are both in the fasta file and potential contaminants

# %%
mask = mq_output.peptides[mq_col.LEADING_RAZOR_PROTEIN].isin(set_proteins_to_remove)
# ToDo: Remove potential contaminants, check evidence.txt
mq_output.peptides.loc[mask, 'Potential contaminant'].value_counts()

# %% [markdown]
# ### `id_map`: Find genes based on fasta file
#
# Using `ID_MAP`, all protein entries for that gene are queried and combined.

# %%
# # slow! discarded for now

# from config import FN_ID_MAP

# with open(FN_ID_MAP) as f:
#     id_map = json.load(f)
# id_map = pd.read_json(FN_ID_MAP, orient="split")

# protein_groups_per_gene = id_map.groupby(by="gene")
# gene_found = []
# for name, gene_data in protein_groups_per_gene:

#     _peptides = set()
#     for protein_id in gene_data.index:
#         _peptides = _peptides.union(p for p_list in data_fasta[protein_id]['peptides']
#                                       for p in p_list)

#     # select intersection of theoretical peptides for gene with observed peptides
#     _matched = mq_output.peptides.index.intersection(_peptides)
#     # add completness?
#     if not _matched.empty and len(_matched) > 3:
#         gene_found.append(name)
#         #
#         if not len(gene_found) % 500 :
#             print(f"Found {len(gene_found):6}")
# print(f"Total: {len(gene_found):5}")

# %% [markdown]
# Compare this with the entries in the `Gene names` column of `peptides.txt`
#
# > Mapping is non-unique. MQ has no treshold on number of identified peptides. (How many (unique) peptides does MQ need?)

# %% [markdown]
# ### `peptides.txt`: Multiple Genes per peptides
#
# - can gene name be collapsed meaningfully?
# - some gene groups share common stem -> can this be used?

# %%
mq_output.peptides[mq_col.GENE_NAMES].head(10)

# %%
gene_sets_unique = mq_output.peptides["Gene names"].unique()

N_GENE_SETS = len(gene_sets_unique)
print(f'There are {N_GENE_SETS} unique sets of genes.')
assert N_GENE_SETS != 0, 'No genes?'

genes_single_unique = mq.get_set_of_genes(gene_sets_unique)
N_GENE_SINGLE_UNIQUE = len(genes_single_unique)

mq.validate_gene_set(N_GENE_SINGLE_UNIQUE, N_GENE_SETS)

# %% [markdown]
# How often do genes names appear in unique sets?

# %%
genes_counted_each_in_unique_sets = pd.Series(mq.count_genes_in_sets(
    gene_sets=gene_sets_unique))

title_ = 'Frequency of counts for each gene in unique set of genes'

ax = genes_counted_each_in_unique_sets.value_counts().sort_index().plot(
    kind='bar',
    title=title_,
    xlabel='Count of a gene',
    ylabel='Frequency of counts',
    ax=None,
)
fig = ax.get_figure()

fig_folder = FIGUREFOLDER / mq_output.folder.stem
fig_folder.mkdir(exist_ok=True)
fig.savefig(fig_folder / f'{title_}.pdf')

# %% [markdown]
# Unique gene sets with more than one gene:

# %%
gene_sets_unique = pd.Series(gene_sets_unique).dropna()

mask_more_than_one_gene = gene_sets_unique.str.contains(';')
gene_sets_unique.loc[mask_more_than_one_gene]

# %% [markdown]
# ### Long format for genes - `peptides_with_single_gene`
#
# Expand the rows for sets of genes using [`pandas.DataFrame.explode`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html).
#
# Does a group of peptide only assigns unique set of genes? Genes can have more than one protein.
#   - first build groups
#   - then see matches (see further below)
#

# %%
peptides_with_single_gene = mq.get_peptides_with_single_gene(
    peptides=mq_output.peptides)
peptides_with_single_gene

# %%
peptides_with_single_gene.dtypes

# %%
print(
    f"DataFrame has due to unfolding now {len(peptides_with_single_gene)} instead of {len(mq_output.peptides)} rows")

# %% [markdown]
# Should peptides from potential contaminants be considered?

# %%
mask = peptides_with_single_gene['Proteins'].str.contains('CON__')
peptides_with_single_gene.loc[mask]

# %%
_mask_con = peptides_with_single_gene.loc[mask, mq_col.PROTEINS].str.split(";").apply(
    lambda x: [True if "CON_" in item else False for item in x]).apply(all)

assert _mask_con.sum() == 0, "There are peptides resulting only from possible confounders: {}".format(
    ", ".join(str(x) for x in peptides_with_single_gene.loc[mask, mq_col.PROTEINS].loc[_mask_con].index))

# %%
peptides_per_gene = peptides_with_single_gene.value_counts(mq_col.GENE_NAMES)
peptides_per_gene

# %% [markdown]
#
# #### Find genes based on `Gene names` column in elonged data-set
#
# More efficient as it does not query unnecessary data or data twice.

# %%
protein_groups_per_gene = peptides_with_single_gene.groupby(
    by=mq_col.GENE_NAMES, dropna=True)

gene_data = protein_groups_per_gene.get_group(peptides_per_gene.index[3])
gene_data

# %%
list_of_proteins = gene_data[mq_col.PROTEINS].str.split(';').to_list()
set_of_proteins = set().union(*list_of_proteins)
set_of_proteins = {x for x in set_of_proteins if 'CON__' not in x}
set_of_proteins

# %%
gene_data[mq_col.PROTEINS].value_counts()  # combine? select first in case of a CON_ as leading razor protein?

# %%
protein_id = set_of_proteins.pop()
print(protein_id)
data_fasta[protein_id]['seq']

# %%
data_fasta[protein_id]

# %% [markdown]
# ### Sample completeness
# Find a sample with a certain completeness level:

# %%
peps_exact_cleaved = mq.find_exact_cleaved_peptides_for_razor_protein(
    gene_data, fasta_db=data_fasta)
peps_exact_cleaved[:10]

# %% [markdown]
# Then search the list of possible peptides originating from the fasta files assuming no miscleavages to the set of found peptides.
#
# - How many unique exact-cleaved peptides can be mapped to any peptide found in the sample (**completness**)?

# %%
peps_in_data = gene_data.index

mq.calculate_completness_for_sample(
    peps_exact_cleaved=peps_exact_cleaved,
    peps_in_data=peps_in_data)

# %% [markdown]
# The number of peptides found can be then used to calculate the completeness

# %% [markdown]
# Select candidates by completeness of training data in single samples and save by experiment name

# %%
mq_output.folder.stem  # needs to go to root?

# %% [markdown]
# ### GeneData accessor?
#
# - [Registering custom accessors tutorial](https://pandas.pydata.org/pandas-docs/stable/development/extending.html#registering-custom-accessors)

# %%
# @pd.api.extensions.register_dataframe_accessor('gene')
# class GeneDataAccessor:

#     COL_INTENSITY  = mq_col.INTENSITY
#     COL_RAZOR_PROT = 'Leading razor protein'
#     COL_PROTEINS   = 'Proteins'
#     COL_GENE_NAME  = 'Gene names'

#     COLS_EXPECTED = {COL_INTENSITY, COL_RAZOR_PROT, COL_PROTEINS, COL_GENE_NAME}

#     def __init__(self, pandas_df):
#         self._validate(df=pandas_df)

#     @classmethod
#     def _validate(cls, df):
#         """Verify if expected columns and layout apply to panda.DataFrame (view)"""
#         _found_columns = cls.COLS_EXPECTED.intersection(df.columns)
#         if not _found_columns == cls.COLS_EXPECTED:
#             raise AttributeError("Expected columns not in DataFrame: {}".format(
#                     list(cls.COLS_EXPECTED - _found_columns)))
#         if not len(df[COL_RAZOR_PROT].unique()) != 1:


# # GeneDataAccessor(gene_data.drop(mq_col.INTENSITY, axis=1))
# # GeneDataAccessor(gene_data)
# # gene_data.drop(mq_col.INTENSITY, axis=1).gene
# gene_data.gene

# %% [markdown]
# ### Gene Data Mapper?

# %%
class GeneDataMapper:

    COL_INTENSITY = mq_col.INTENSITY
    COL_RAZOR_PROT = mq_col.LEADING_RAZOR_PROTEIN
    COL_PROTEINS = mq_col.PROTEINS
    COL_GENE_NAME = mq_col.GENE_NAMES

    COLS_EXPECTED = {COL_INTENSITY, COL_RAZOR_PROT,
                     COL_PROTEINS, COL_GENE_NAME}

    def __init__(self, pandas_df, fasta_dict):
        self._validate(df=pandas_df)
        self._df = pandas_df
        self._fasta_dict = fasta_dict

        # # self.log?

    @classmethod
    def _validate(cls, df):
        """Verify if expected columns and layout apply to panda.DataFrame (view)"""
        _found_columns = cls.COLS_EXPECTED.intersection(df.columns)
        if not _found_columns == cls.COLS_EXPECTED:
            raise AttributeError("Expected columns not in DataFrame: {}".format(
                list(cls.COLS_EXPECTED - _found_columns)))
        if len(df[cls.COL_RAZOR_PROT].unique()) != 1:
            raise ValueError(
                "Non-unique razor-protein in DataFrame: ", df[cls.COL_RAZOR_PROT].unique())

    def __repr__(self):
        return f"{self.__class__.__name__} at {id(self)}"


GeneDataMapper(gene_data, data_fasta)

# %% [markdown]
# ### Dump samples as json
#
# - select unique gene-names in set (have to be shared by all peptides)
# - dump peptide intensities as json from `peptides.txt`

# %%
peptides_with_single_gene  # long-format with repeated peptide information by gene

# %%
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.handlers

# %%
genes_counted_each_in_unique_sets = pd.Series(mq.count_genes_in_sets(
    gene_sets=gene_sets_unique))

# # ToDo: Develop
# class MaxQuantTrainingDataExtractor():
#     """Class to extract training data from `MaxQuantOutput`."""

#     def __init__(self, out_folder):
#         self.out_folder = Path(out_folder)
#         self.out_folder.mkdir(exist_ok=True)
#         self.fname_template = '{gene}.json'

completeness_per_gene = mq.ExtractFromPeptidesTxt(
    out_folder='train', mq_output_object=mq_output, fasta_db=data_fasta)()

# %%
# same code fails in `vaep.io.mq`, ABC needed?
isinstance(mq_output, MaxQuantOutput), type(mq_output)

# %% [markdown]
# #### Descriptics

# %%
s_completeness = pd.Series(completeness_per_gene, name='completenes_by_gene')
s_completeness.describe()

# %%
N_BINS = 20
ax = s_completeness.plot(
    kind='hist',
    bins=N_BINS,
    xticks=[
        x /
        100 for x in range(
            0,
            101,
            5)],
    figsize=(
        10,
        5),
    rot=90,
    title=f"Frequency of proportion of observed exact peptides (completness) per razor protein from 0 to 1 in {N_BINS} bins"
    f"\nin sample {mq_output.folder.stem}")

_ = ax.set_xlabel(
    "Proportion of exactly observed peptides (including up to 2 mis-cleavages)")

fig = ax.get_figure()
fig.tight_layout()
fig.savefig(FIGUREFOLDER / mq_output.folder.stem / 'freq_completeness.png')

# %% [markdown]
# based on completeness, select valid training data

# %%
# continously decrease this number in the scope of the project
mask = s_completeness > .6
s_completeness.loc[mask]
