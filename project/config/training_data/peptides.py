from ..counter_fpaths import FNAME_C_PEPTIDES
from vaep.io import data_objects

NAME = 'peptides'
BASE_NAME = f"df_intensities_{NAME}_long"

TYPES_DUMP = {'Sample ID': 'category',
                  'Sequence': 'category',
                  }

TYPES_COUNT = {}

IDX_COLS_LONG = ['Sample ID', 'Sequence'] # in order 

LOAD_DUMP = data_objects.load_agg_peptide_dump

CounterClass = data_objects.PeptideCounter
FNAME_COUNTER = FNAME_C_PEPTIDES

