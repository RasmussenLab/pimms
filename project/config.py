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

PROTEIN_DUMPS = Path(PROCESSED_DATA) / 'processed'
os.makedirs(PROTEIN_DUMPS, exist_ok=True)


#FN_PEPTIDE_INTENSITIES = Path(FOLDER_DATA) / 'mq_out' / 'peptide_intensities.pkl'
FN_PEPTIDE_INTENSITIES = Path(FOLDER_DATA) / 'peptide_intensities.pkl'

PREFIX_IMPUTED = 'hela_imputed_proteins'
PREFIX_META = 'hela_metadata'

FOLDER_FASTA = Path(FOLDER_DATA) / 'fasta'
FN_FASTA_DB = FOLDER_FASTA / 'fasta_db.json'
FN_ID_MAP = FOLDER_FASTA / 'id_map.json'
FN_PROT_GENE_MAP = FOLDER_FASTA / 'uniprot_protein_gene_map.json'
FN_PEP_TO_PROT = FOLDER_FASTA / 'peptided_to_prot_id.json'
FN_PROTEIN_SUPPORT_MAP = Path(FOLDER_DATA) / 'protein_support.pkl'
FN_PROTEIN_SUPPORT_FREQ = Path(FOLDER_DATA) / 'dict_protein_support_freq.pkl'

# DATA FASTA Config
KEY_FASTA_HEADER = 'meta'
KEY_FASTA_SEQ = 'seq'
KEY_PEPTIDES = 'peptides'
KEY_GENE_NAME = 'gene'
KEY_GENE_NAME_FASTA = 'gene_fasta'