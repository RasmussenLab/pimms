"""
Project config file.

Different config for different settings.

os to pathlib functionaly, see
https://docs.python.org/3/library/pathlib.html#correspondence-to-tools-in-the-os-module

"""
import os
from collections import namedtuple
from pathlib import Path

def mkdir(path=Path): 
    path.mkdir(exist_ok=True)
    return path

# project folder specific
FIGUREFOLDER = Path('Figures')
FIGUREFOLDER.mkdir(exist_ok=True)

FOLDER_DATA = Path('data')
FOLDER_DATA.mkdir(exist_ok=True)

FOLDER_PROCESSED = FOLDER_DATA / 'processed'
FOLDER_PROCESSED.mkdir(exist_ok=True)

FOLDER_TRAINING = mkdir(FOLDER_DATA / 'hela_qc_data')

# (old) Synonyms 
PROCESSED_DATA = FOLDER_PROCESSED
PROTEIN_DUMPS = PROCESSED_DATA

####
####

#local PC config
# FOLDER_MQ_TXT_DATA = Path('data') / 'mq_out/'
# FOLDER_KEY  = None

# erda specific
FOLDER_MQ_TXT_DATA = Path('/home/jovyan/work/mq_out/')
FOLDER_RAW_DATA = Path('/home/jovyan/work/Hela/')
FOLDER_KEY = 'txt'
 
#FN_PEPTIDE_INTENSITIES = Path(FOLDER_DATA) / 'mq_out' / 'peptide_intensities.pkl'
FN_PEPTIDE_STUMP = 'peptide_intensities'
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

KEYS_FASTA_ENTRY = [KEY_FASTA_HEADER, KEY_FASTA_SEQ, KEY_PEPTIDES, KEY_GENE_NAME]

FastaEntry = namedtuple('FastaEntry', KEYS_FASTA_ENTRY)
fasta_entry = FastaEntry(*KEYS_FASTA_ENTRY)


FILEPATH_UTILS = 'src/file_utils.py'