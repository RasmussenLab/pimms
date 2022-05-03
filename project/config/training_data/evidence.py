from ..counter_fpaths import FNAME_C_EVIDENCE
from vaep.io import data_objects

NAME = 'evidence'
BASE_NAME = f"df_intensities_{NAME}_long"

TYPES_DUMP = {'Sample ID': 'category',
              'Sequence': 'category',
              'Charge': 'category',}

TYPES_COUNT = {'Charge': int}

IDX_COLS_LONG = ['Sample ID', 'Sequence', 'Charge'] # in order 

LOAD_DUMP = data_objects.load_evidence_dump

CounterClass = data_objects.EvidenceCounter
FNAME_COUNTER = FNAME_C_EVIDENCE