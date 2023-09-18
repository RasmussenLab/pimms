# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: vaep
#     language: python
#     name: vaep
# ---

# %% [markdown]
# # Process FASTA files
# > uses only the provided fasta files in `src.config.py` by `FOLDER_FASTA`
#
# - create theoretically considered peptides considered by search engines
# - dump results as human readable json to `FN_FASTA_DB` file specifed in src.config.
#
# > Based on notebook received by [Annelaura Bach](https://www.cpr.ku.dk/staff/mann-group/?pure=en/persons/443836) and created by Johannes B. Müller \[[scholar](https://scholar.google.com/citations?user=Rn1OS8oAAAAJ&hl=de), [MPI Biochemistry](https://www.biochem.mpg.de/person/93696/2253)\]

# %%

from collections import defaultdict, namedtuple
import os
import json
import logging
from pathlib import Path

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# %%
from vaep.fasta import cleave_to_tryptic
from vaep.fasta import iterFlatten
from vaep.fasta import count_peptide_matches
from vaep.fasta import read_fasta
from vaep.io import search_files
from vaep.pandas import combine_value_counts
from vaep.databases.uniprot import query_uniprot_id_mapping
from vaep.utils import sample_iterable
from vaep.plotting import _savefig

# %%
from config import FN_FASTA_DB
from config import FIGUREFOLDER
from config import FN_ID_MAP
from config import FN_PROT_GENE_MAP
from config import FN_PEP_TO_PROT

from config import KEY_FASTA_HEADER, KEY_FASTA_SEQ, KEY_GENE_NAME, KEY_PEPTIDES

# %% [markdown]
# ## Core Functionality - Example
#
# - write tests for core functinality
# - refactor to file

# %%
test_data = {
    "meta": ">tr|A0A024R1R8|A0A024R1R8_HUMAN HCG2014768, isoform CRA_a OS=Homo sapiens OX=9606 GN=hCG_2014768 PE=4 SV=1",
    "seq": "MSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK",
    "peptides": [
        "MSSHEGGK",
        "EMDEEEK",
        "GPLATGGIK"],
}

# %% [markdown]
# regex is slower than native string replacing and splitting in Python

# %%
# import re
# cut_by_trypsin = re.compile('([^K]+K)|([^R]+R)')
# _res = cut_by_trypsin.split(test_data['seq'])
# [_pep for _pep in _res if _pep != '' and _pep != None]

# %% [markdown]
#
# - map peptide set of peptides (how to deal with mis-cleavages?)
#     - mis-cleavages can happen both to the peptide before and after.
#     > `pep1, pep2, pep3, pep4, pep5`
#     > `pep1pep2, pep2pep3, pep3pep4, pep4pep5`
#     - sliding windows can pass trough the list of peptides - should work with recursion

# %%
l_peptides = test_data["seq"].replace("K", "K ").replace("R", "R ").split()
l_peptides

# %% [markdown]
# `add_rxk` should add pattern of starting R and trailing K ?

# %%
last_pep = ""
temp_peps = []
num_missed_cleavages = 1
add_rxk = True

sec_last_pep = ""

pep_rdx = []

for pep in l_peptides:
    if last_pep != "":
        temp_peps.append(last_pep + pep)
    if add_rxk and sec_last_pep != "" and len(sec_last_pep) <= 2:
        _pep_rxk = sec_last_pep + last_pep + pep
        print(_pep_rxk)
        pep_rdx.append(_pep_rxk)
        temp_peps.append(_pep_rxk)

    sec_last_pep = last_pep  # sec_last_pep, last_pep = last_pep, pep ?
    last_pep = pep
temp_peps

# %%
repr(pep_rdx)

# %% [markdown]
# Missed cleavages core functionality (adapted)

# %%
example_peptides_fasta = cleave_to_tryptic(
    test_data["seq"], num_missed_cleavages=2, add_rxk=True
)
print("number of peptides: ", [len(_l) for _l in example_peptides_fasta])
example_peptides_fasta[-1]

# %%
print("".join(example_peptides_fasta[0]), *example_peptides_fasta, sep="\n")

# %% [markdown]
# rdx peptides are a subset of two missed cleavage sites peptides. There
# are omitted when two and more cleavage site can be skipped.

# %%
example_peptides_fasta = cleave_to_tryptic(
    test_data["seq"], num_missed_cleavages=1, add_rxk=True
)
print("number of peptides: ", [len(_l) for _l in example_peptides_fasta])
example_peptides_fasta[-1]

# %% [markdown]
# Data Structure is no a list of list. Maybe this could be improved.
# Information what kind of type the peptide is from, is still interesting.

# %% [markdown]
# ## Process Fasta Files
#
# First define input Folder and the file location of the created peptides:

# %%
fasta_files = search_files(path=".", query=".fasta")
print(fasta_files)
print("\n".join(fasta_files.files))

# %% [markdown]
# ### Define Setup
#
# Set input FASTA, Output .txt name, lower legth cutoff, missed cleavages and if to report reverse.
#
# Tryptic digest of Fastas to Peptides >6 in list for matching with measured peptides

# %%
CUTOFF_LEN_PEP = 7
MAX_MISSED_CLEAVAGES = 2  # default in MaxQuant
DECOY_REVERSE = False
SUMMARY_FILE = "tex/fasta_tryptic_analysis.tex"

_summary_text = (
    "The theoretical analysis of the fasta files gives an idea about how many possible peptides \n"
    "can be expected by cleaving proteins using trypsin. The hyperparameters for peptide creation are \n"
    f"to consider the minimal peptide length to be {CUTOFF_LEN_PEP} amino acids, \n"
    f"to consider a maximum of {MAX_MISSED_CLEAVAGES} missed cleavage sites (default in MaxQuant) and \n"
    f"to {'not ' if not DECOY_REVERSE else ''}add decoy peptides by reversing peptide sequences. \n"
)
print(_summary_text, sep="\n")

# %% [markdown]
# From the [Fasta Meta information](https://ebi14.uniprot.org/help/fasta-headers) the Identifier is extracted.
#
# ```
# >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
# ```
# - db is `sp` for UniProtKB/Swiss-Prot and `tr` for UniProtKB/TrEMBL.
# - `UniqueIdentifier` is the primary *accession number* of the UniProtKB entry. (seems to be used by MQ)
# - `EntryName` is the entry name of the UniProtKB entry.
# - `ProteinName` is the recommended name of the UniProtKB entry as annotated in the *RecName* field. For UniProtKB/TrEMBL entries without a *RecName* field, the *SubName* field is used. In case of multiple SubNames, the first one is used. The 'precursor' attribute is excluded, 'Fragment' is included with the name if applicable.

# %% [markdown]
# `>tr` or `>sp`

# %% [markdown]
# ### Schema for single fasta entry

# %%


data_fasta = {}

# # add Schema?
# schema_fasta_entry = {
#                       KEY_FASTA_HEADER: str,
#                       KEY_GENE_NAME: str,
#                       KEY_FASTA_SEQ: str,
#                       KEY_PEPTIDES: (list, (2,2))
#                      }
# # or dataclass
# from dataclasses import make_dataclass
# FastaEntry = make_dataclass(cls_name='FastaEntry',
#                             fields=[
#                                 (KEY_FASTA_HEADER, 'str'),
#                                 (KEY_GENE_NAME, 'str'),
#                                 (KEY_FASTA_SEQ, 'str'),
#                                 (KEY_PEPTIDES, list)
#                             ])
# # or namedtuple
# FastaEntry = namedtuple('FastaEntry', [KEY_FASTA_HEADER, KEY_GENE_NAME, KEY_FASTA_SEQ, KEY_PEPTIDES])

# %% [markdown]
# How to validate schema of fasta entry stored as dictionary?
# - [schema](https://stackoverflow.com/questions/45812387/how-to-validate-structure-or-schema-of-dictionary-in-python) validation in python discussion

# %% [markdown]
# ### Process Fasta file

# %%
for _fasta in tqdm(fasta_files.files):

    with open(_fasta) as fp:
        for i, (metainfo, seq) in tqdm(enumerate(read_fasta(fp))):
            identifier = metainfo.split("|")[1]
            gene = "|".join([x.split("=")[-1] for x in metainfo.split() if "GN=" in x])
            if identifier in data_fasta:
                raise ValueError("Key seen before: {}".format(identifier))
            _all_peptides = cleave_to_tryptic(
                seq, num_missed_cleavages=MAX_MISSED_CLEAVAGES, reversed=DECOY_REVERSE
            )
            data_fasta[identifier] = {
                KEY_FASTA_HEADER: metainfo,
                KEY_GENE_NAME: gene,
                KEY_FASTA_SEQ: seq,
                KEY_PEPTIDES: [
                    [_pep for _pep in _peptides if len(_pep) >= CUTOFF_LEN_PEP]
                    for _peptides in _all_peptides
                ],
            }

# %% [markdown]
# `fasta_data` holds all information to pick a subset of peptides from peptides intensity tables

# %%
# from random import sample
# sample_ids = sample(list(data_fasta), 10)
# for _id in sample_ids:
#     print("Unique Identifier: {}: \n\t AA-Seq: {} \n\t Header: {} \n\t Peptides: {}\n".format(_id, data_fasta[_id]['seq'], data_fasta[_id]['meta'], data_fasta[_id]['peptides']))
data_fasta["A0A024R1R8"]

# %%
d_seq_length = {}
for _key, _data in data_fasta.items():
    d_seq_length[_key] = len(_data[KEY_FASTA_SEQ])

# %%
d_seq_length = pd.Series(d_seq_length)
d_seq_length.describe()

# %%
test_series = pd.Series({"A": 4, "B": 1, "C": 0, "D": 4})


def get_indices_with_value(s: pd.Series, value):
    """Return indices for with the value is true"""
    return s[s == value].index


get_indices_with_value(test_series, 4)

# %% [markdown]
# Boolean Indexing, remember to set
# [parantheses](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing)

# %%
MIN_AA_IN_SEQ = 10
MAX_AA_IN_SEQ = 2000
mask_min = d_seq_length < MIN_AA_IN_SEQ
mask_max = d_seq_length > MAX_AA_IN_SEQ
# _summary_text += f"\nThe FASTA file contain {sum(mask_min)} proteins with less than {MIN_AA_IN_SEQ} amino acids (AAs) and {sum(mask_max)} with more than {MAX_AA_IN_SEQ} AAs."
_summary_text += (
    f"The minimal AA sequence length is {min(d_seq_length)} of UniProt ID {', '.join(get_indices_with_value(d_seq_length, min(d_seq_length)))} "
    f"and the maximal sequence lenght is {max(d_seq_length)} for UniProt ID {', '.join(get_indices_with_value(d_seq_length, max(d_seq_length)))}"
)
print(_summary_text)

# %%
_ = d_seq_length.loc[(~mask_max)].to_frame(name="AA Seq Length").plot.hist(bins=200)

# %%
l_genes = []
n_set = 0
for _key, _data in data_fasta.items():
    _gene_name = _data[KEY_GENE_NAME]
    if _gene_name:
        l_genes.append(_gene_name)
        n_set += 1

# %%
_summary_text += (
    f"\nIn the FASTA header file {n_set} proteins have a set gene of a total of {len(data_fasta)} proteins,"
    f" i.e. {len(data_fasta) - n_set} have an undefined origin. There are {len(set(l_genes))} unique gene names in the FASTA file specified.\n"
)
print(_summary_text)

# %%
len(set(l_genes))

# %% [markdown]
# ## Number of well-defined peptides per protein (isotope)
#
# - well-defined peptides := no cleavage site is missed

# %%
peps_exact_count_freq = defaultdict(int)

for key, d_data in data_fasta.items():
    _N = len(d_data[KEY_PEPTIDES][0])
    # if _N == 0:
    #     print(key)
    #     print(d_data)
    peps_exact_count_freq[_N] += 1
peps_exact_count_freq = pd.Series(dict(peps_exact_count_freq)).sort_index()
peps_exact_count_freq

# %%
n_first = 40
ax = peps_exact_count_freq.iloc[:n_first].plot(kind='bar',
                                               figsize=(20, 5),
                                               title=f'Frequency of number of exact peptides (up to {peps_exact_count_freq.iloc[:40].index[-1]})'
                                               f' representing {peps_exact_count_freq.iloc[:40].sum()} proteins out of '
                                               f'{peps_exact_count_freq.sum()} ({peps_exact_count_freq.iloc[:40].sum()/peps_exact_count_freq.sum():.2f}%)',
                                               xlabel="Number of exact peptides (considered) in protein sequence",
                                               ylabel="Number of protein(s) (incl. isotopes)",
                                               fontsize=10)

# %%
peps_exact_count_freq = pd.Series(dict(peps_exact_count_freq)).sort_index()
fig = ax.get_figure()
fig.savefig(Path(FIGUREFOLDER) / 'fasta_exact_peptide_count_freq.png')
fig.savefig(Path(FIGUREFOLDER) / 'fasta_exact_peptide_count_freq.pdf')

# %% [markdown]
# ### Proteins' Isoforms

# %% [markdown]
# Possible to join "isoforms" by joining all variants to one. Isoforms are
# numbered from the second on by appending `-i` for $i>1$, i.e. starting
# with `-2`. The gene name of which the protein (isoform) originate can be
# obtained by using [id
# mapping](https://www.uniprot.org/help/api_idmapping). Isoforms are not
# mapped automatically by Uniprot to its GENENAME, i.e. you have to strip
# all `-i`, e.g `-2`, `-3`, for querying. Here the protein, gene pairs are
# mapped to the unique protein identifiers.

# %%
prot_ids = list(data_fasta.keys())
prot_ids = pd.Series(prot_ids)
prot_ids

# %%
mask = prot_ids.str.contains("-")
isoforms = prot_ids.copy().loc[mask]
isoforms

# %%
N_prot_with_isoform = isoforms.str.split("-").str[0].nunique()

# %%
n_unique_proteins_wo_isoforms = len(prot_ids) - len(isoforms)
_summary_text += "\nA total of {} proteins have at least one more isoform. ".format(
    N_prot_with_isoform
)
_summary_text += f"Collapsing isoforms into one protein results in {n_unique_proteins_wo_isoforms} proteins."
print(_summary_text)

# %% [markdown]
# Remove Isoforms from list. How to integrate this information before?
#
# fasta-data has to be merge one-to-many.

# %%
id_map = pd.DataFrame(
    prot_ids.str.split("-").str[0], columns=["protein"]
)  # , index=list(prot_ids))
id_map.index = pd.Index(prot_ids, name="prot_id")
id_map

# %%
id_map.loc[id_map.index.str.contains("A0A096LP49|Q9Y6Z5|W5XKT8")]

# %%
l_proteins = id_map.protein.unique()
print(
    f"There are {len(l_proteins)} unique proteins without isoforms listed in the used fasta files."
)
# Check with pervious result.
assert n_unique_proteins_wo_isoforms == len(l_proteins)

# %%
try:
    with open(FN_PROT_GENE_MAP) as f:
        dict_protein_to_gene = json.load(f)
    logging.warning(f"Loaded pre-cached map dict_protein_to_gene: {FN_PROT_GENE_MAP}")
except FileNotFoundError:
    dict_protein_to_gene = {}
    start = 0
    for end in list(range(10000, len(l_proteins), 10000)):
        print(f"Retrieve items {start+1:6} to {end:6}")
        _id_to_gene = query_uniprot_id_mapping(l_proteins[start:end])
        print(f"Found {len(_id_to_gene)} gene names")
        dict_protein_to_gene.update(_id_to_gene)
        start = end
    print(f"Retrieve items {start:6} to {len(l_proteins):6}")
    _id_to_gene = query_uniprot_id_mapping(l_proteins[start:])
    print(f"Found {len(_id_to_gene)} gene names")
    dict_protein_to_gene.update(_id_to_gene)
    with open(FN_PROT_GENE_MAP, "w") as f:
        json.dump(dict_protein_to_gene, f, indent=4, sort_keys=False)

# %%
genes = pd.Series(dict_protein_to_gene, name="gene")
genes

# %%
assert (
    len(genes) == 72471
), f"The number of proteins associated to a gene found on 11.11.2020 was 72471, now it's {len(genes)}"

# %% [markdown]
# Add gene names from UniProt to `id_map` DataFrame by an outer join
# (keeping all information based on the protein names shared by isotopes)

# %%
id_map = id_map.merge(genes, how="outer", left_on="protein", right_index=True)
id_map.sort_values(by=["gene", "protein"], inplace=True)
id_map

# %%
id_map.replace('', np.nan)

# %% [markdown]
# Add the gene name collected previously from the Fasta Header

# %%
genes_fasta_offline = pd.DataFrame(
    ((_key, _data[KEY_GENE_NAME]) for _key, _data in data_fasta.items()),
    columns=["prot_id", "gene_fasta"],
).set_index("prot_id"
            ).replace('', np.nan)
genes_fasta_offline.loc[genes_fasta_offline.gene_fasta.isna()]

# %%
id_map = id_map.merge(
    genes_fasta_offline,
    how="outer",
    left_index=True,
    right_index=True)
id_map.sort_values(by=["gene", "protein"], inplace=True)
id_map

# %%
mask_no_gene = id_map.gene.isna()
id_map.loc[mask_no_gene]

# %% [markdown]
# Using the genes from the fasta file header reduces the number of missing
# genes, but additionally other differences arise in the comparison to the
# lastest version.

# %%
mask_gene_diffs = id_map.gene != id_map.gene_fasta
id_map.loc[mask_gene_diffs]

# %%
id_map.gene.isna().sum(), id_map.gene_fasta.isna()

# %%
id_map.loc[(id_map.gene.isna()) & (id_map.gene_fasta.isna())]

# %%
_summary_text += (
    f"\nThere are {id_map.gene.isna().sum()} protein IDs (or {id_map.loc[mask_no_gene].protein.nunique()} proteins) "
    "without a gene associated to them in the current online version of UniProt, "
    f"whereas there are no genes for only {id_map.gene_fasta.isna().sum()} in the headers of proteins in the used FASTA files."
)
print(_summary_text)

# %% [markdown]
# ### Isotopes mapping
#
# Isotopes are mapped now to a protein with the same name. The same can be
# achieved by just discarding everything behind the hypen `-`

# %%
id_map.loc[id_map.index.str.contains("-")]

# %% [markdown]
# Save id_map

# %%
id_map.to_json(FN_ID_MAP, orient="split", indent=4)

# %% [markdown]
# ### Most proteins with a missing gene are deleted
#
# If one checks manually some of the examples (e.g. the hundred provided here), one sees that all are deleted from Uniprot.
#
# > How to obtain different versions of UniProt?!

# %%
if not len(dict_protein_to_gene) == len(l_proteins):
    print("Not all ids are mapped.")
    _diff = set(l_proteins).difference(dict_protein_to_gene.keys())
    print(f"Number of protein identifiers not mapped to a gene in UniProt online: {len(_diff)}")
    print(f'Look at {100} examples: {", ".join(sample_iterable(_diff, 100))}')

# %%
_summary_text += (
    f"\nMost of the {len(_diff)} proteins ({len(_diff)/len(l_proteins)*100:.2f} percent of the unique proteins) "
    "not mapped to a gene name are deleted in the most current version of UniProt (online). "
    "The versioning of the fasta-files has to be investigated, as there arise differences over time due to updates."
)
_summary_text += (
    f"\nProteins are mapped to a total number of genes of {id_map.gene.nunique()} in the online UniProt version and {id_map.gene_fasta.nunique()} in the offline used one.\n"
)
print(_summary_text)

# %%
f"Proteins are mapped to a total number of genes of {len(set(dict_protein_to_gene.values()))}"

# %% [markdown]
# ### Map peptide to either identifier, common protein or gene
#

# %%
peptide_to_prot = defaultdict(list)
for _id, _data in tqdm(data_fasta.items()):
    for _pep in iterFlatten(_data["peptides"]):
        peptide_to_prot[_pep].append(_id)

_summary_text += f"\nConsidering {MAX_MISSED_CLEAVAGES} missed cleavage site(s) there are {len(peptide_to_prot):,d} unique peptides."

# %%
print(_summary_text)

# %%
{_key: peptide_to_prot[_key] for _key in sample_iterable(peptide_to_prot.keys())}

# %%
# %%time
with open(FN_PEP_TO_PROT, "w") as f:
    json.dump(peptide_to_prot, f, indent=4, sort_keys=False)

# %% [markdown]
# ### Plot histograms for different levels of abstraction
#
# Plot counts of matched
#    1. protein IDs
#    2. proteins (joining isoforms)
#    3. genes
#
# to their peptides. See how many unique peptides exist. The number of
# peptides should stay the same, so the counts do not have to be
# normalized.

# %%
USE_OFFLINE_FASTA_GENES = True
if USE_OFFLINE_FASTA_GENES:
    dict_protein_to_gene = genes_fasta_offline.loc[~genes_fasta_offline.index.str.contains('-')]
    dict_protein_to_gene = dict_protein_to_gene.dropna().to_dict()['gene_fasta']

# %%
{_key: dict_protein_to_gene[_key] for _key in sample_iterable(dict_protein_to_gene.keys(), 10)}

# %%
len(dict_protein_to_gene)

# %%
counters = {}
levels = ["protein_id", "protein", "gene"]
for level in levels:
    counters[level] = pd.Series(
        count_peptide_matches(peptide_to_prot, dict_protein_to_gene, level=level)
    )

# %%
for level in levels:
    print(f"{level}: {counters[level]['AACLCFR']}")

# %%
peptide_to_prot["AACLCFR"]

# %%
_prots = {x.split("-")[0] for x in peptide_to_prot["AACLCFR"]}
{dict_protein_to_gene[_prot] for _prot in _prots}

# %%
counts_by_level = combine_value_counts(pd.DataFrame(counters))
counts_by_level = counts_by_level.replace(np.nan, 0).astype(int)
counts_by_level

# %% [markdown]
# Interpretation: Peptides are assigned \# of times to a protein_id, protein or gene respectively.

# %% [markdown]
# Check that for all levels the same number of peptides are counted.

# %%
counts_by_level.sum()

# %% [markdown]
# Plot the frequency of matched proteins to one peptide sequence:

# %%
fig, ax = plt.subplots(figsize=(13, 7))

ax = counts_by_level.iloc[:5].plot(kind="bar", ax=ax)
ax.set_ylabel("peptide counts")
ax.set_xlabel("number of matched levels")
# ax.yaxis.set_major_formatter("{x:,}")
_y_ticks = ax.set_yticks(list(range(0, 3_500_000, 500_000)))  # is there a ways to transform float to int in matplotlib?
_y_ticks_labels = ax.set_yticklabels([f"{x:,}" for x in range(0, 3_500_000, 500_000)])

_savefig(fig, folder="figures", name="fasta_top4")

# %%
fig, axes = plt.subplots(2, 2, figsize=(17, 10))
axes = axes.flatten()

counts_by_level.iloc[:10].plot(kind="bar", ax=axes[0])
axes[0].set_title("up to 9 matches")
axes[0].set_yticks(list(range(0, 3_500_000, 500_000)))
axes[0].set_yticklabels(['0', '500,000', '1,000,000', '1,500,000', '2,000,000', '2,500,000', '3,000,000'])

_start = 10
for i, _end in enumerate([31, 61], start=1):
    counts_by_level.iloc[_start:_end].plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{_start} to {_end-1} matches")
    _start = _end

i += 1
counts_by_level.iloc[-30:].plot(kind="bar", ax=axes[i])
axes[i].set_title(f"{30} most frequent matches")


axes = axes.reshape((2, 2))

pad = 5  # in point
for i in range(2):
    axes[-1, i].set_xlabel("Count of number of matches for a peptide")
    axes[i, 0].set_ylabel("number of peptides")

_ = fig.suptitle(
    "Frequency of peptides matched to x items of protein IDs, proteins (combining isotopes) and genes",
    fontsize=16,
)


fig.tight_layout()
_savefig(fig, folder="figures", name="fasta_mapping_counts")

# %% [markdown]
# check for homology of sequences in python?

# %% [markdown]
# ## Create Automated report
#
# - paragraph in tex for report
# - summary table

# %%
print(_summary_text)

# %%
Path(SUMMARY_FILE).parent.mkdir(exist_ok=True)
with open(Path(SUMMARY_FILE), "w") as f:
    f.write(_summary_text)

# %% [markdown]
# ## Save mappings as JSON
#
# Each `protein_id` is an entry with the following information:
# ```
# 'meta': <fasta-header>
# 'seq': <protein-sequence>
# 'peptides': <list of list of peptide sequences: [[0-missed-cleavages, 1-missed-cleavage, 2-missed-cleavage]]>
# ```

# %%
# %%time
with open(FN_FASTA_DB, "w") as f:
    json.dump(data_fasta, f, indent=4, sort_keys=False)

# %%
os.stat(FN_FASTA_DB).st_size / 1024 / 1024
