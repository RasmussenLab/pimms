import os
from pathlib import Path

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
PREFIX_META = 'hela_metadata'

FOLDER_FASTA = Path(FOLDER_DATA) / 'fasta'
FN_FASTA_DB = FOLDER_FASTA / 'fasta_db.json'
FN_ID_MAP = FOLDER_FASTA / 'id_map.json'
FN_PROT_GENE_MAP = FOLDER_FASTA / 'uniprot_protein_gene_map.json'
FN_PEP_TO_PROT = FOLDER_FASTA / 'peptided_to_prot_id.json'
