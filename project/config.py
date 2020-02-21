import os

FILENAME = 'Mann_Hepa_data.tsv'

FIGUREFOLDER = 'Figures'
os.makedirs(FIGUREFOLDER, exist_ok=True)

DATAFOLDER = 'data'

PROCESSED_DATA = os.path.join(DATAFOLDER, 'processed')
os.makedirs(PROCESSED_DATA, exist_ok=True)


PREFIX_IMPUTED = 'hela_imputed_proteins'
PREFIX_META    = 'hela_metadata'