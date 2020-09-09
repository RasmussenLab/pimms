import os

FILENAME = 'Mann_Hepa_data.tsv'

FIGUREFOLDER = 'Figures'
os.makedirs(FIGUREFOLDER, exist_ok=True)

FOLDER_RAW_DATA = '/home/jovyan/work/mq_out/'
FOLDER_KEY  = None

# FOLDER_RAW_DATA = '/home/jovyan/work/Hela/'
# FOLDER_KEY = 'txt'
FOLDER_RAW_DATA = os.path.abspath(FOLDER_RAW_DATA)
FOLDER_DATA = 'data'

PROCESSED_DATA = os.path.join(FOLDER_DATA, 'processed')
os.makedirs(PROCESSED_DATA, exist_ok=True)


PREFIX_IMPUTED = 'hela_imputed_proteins'
PREFIX_META    = 'hela_metadata'