from ..counter_fpaths import FNAME_C_GENES # use genes as identifier between samples
from vaep.io import data_objects

NAME = 'proteinGroups'

BASE_NAME = f"df_intensities_{NAME}_long"

TYPES_DUMP = {'Sample ID': 'category',
              'Gene names': 'category',
              }

TYPES_COUNT = {}

IDX_COLS_LONG = ['Sample ID', 'Gene names']  # in order

LOAD_DUMP = data_objects.pg_idx_gene_fct

CounterClass = data_objects.GeneCounter

FNAME_COUNTER = FNAME_C_GENES
