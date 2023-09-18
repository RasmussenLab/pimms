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

# %%
from config import FNAME_C_PEPTIDES, FNAME_C_EVIDENCE, FNAME_C_PG, FNAME_C_GENES
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vaep
from vaep.io import data_objects
from vaep.logging import setup_nb_logger
setup_nb_logger(level=logging.INFO)


FNAME_C_PEPTIDES, FNAME_C_EVIDENCE, FNAME_C_PG, FNAME_C_GENES

# %% [markdown]
# ## Aggregated Peptides

# %%
peptide_counter = data_objects.PeptideCounter(FNAME_C_PEPTIDES)
N_SAMPLES = len(peptide_counter.loaded)

# %%
peptide_counter

# %%
peptide_counts = peptide_counter.get_df_counts()
# peptide_counts.index += 1
peptide_counts.head()

# %%
peptide_counts.describe(percentiles=np.linspace(0.1, 1, 10))

# %%
vaep.plotting.make_large_descriptors()
ax = peptide_counter.plot_counts()

# %% [markdown]
# ## Evidence - Peptides by charge and modifications
#
#

# %%
evidence_counter = data_objects.EvidenceCounter(FNAME_C_EVIDENCE)
evidence_count = evidence_counter.get_df_counts()
evidence_count.head()

# %%
ax = evidence_counter.plot_counts()

# %% [markdown]
# ## Protein Groups

# %%
pg_counter = data_objects.ProteinGroupsCounter(FNAME_C_PG)
pg_count = pg_counter.get_df_counts()
pg_count.head()

# %%
ax = pg_counter.plot_counts()

# %% [markdown]
# ## Genes

# %%
gene_counter = data_objects.GeneCounter(FNAME_C_GENES)
gene_count = gene_counter.get_df_counts()
gene_count.head()  # remove NaN entry


# %%
gene_count = gene_count.iloc[1:]
gene_count.head()

# %%
ax = gene_counter.plot_counts(df_counts=gene_count)  # provide manuelly manipulated gene counts
