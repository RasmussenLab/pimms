# src.config goes here
# import src.config -> import config

"""
Project config file.

Different config for different settings.

os to pathlib functionaly, see
https://docs.python.org/3/library/pathlib.html#correspondence-to-tools-in-the-os-module

"""
import vaep.io
import logging
import os
import yaml
from collections import namedtuple
from pathlib import Path, PurePath, PurePosixPath
from pprint import pformat

import numpy as np
import pandas
import matplotlib as mpl


def mkdir(path=Path):
    path.mkdir(exist_ok=True, parents=True)
    return path


logger = logging.getLogger('vaep')

###############################################################################
###############################################################################
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

###############################################################################
###############################################################################
# Adapt this part
ON_ERDA = True
#local PC config
FOLDER_MQ_TXT_DATA = None

FOLDERS_MQ_TXT_DATA = [
    Path('Y:/') / 'mq_out',
    FOLDER_DATA / 'mq_out',
    Path('/home/jovyan/work/mq_out/'),
]
for folder in FOLDERS_MQ_TXT_DATA[:-1]:
    if folder.exists():
        print(f'FOLDER_MQ_TXT_DATA = {folder}')
        FOLDER_MQ_TXT_DATA = folder
        ON_ERDA = False
        break

if FOLDERS_MQ_TXT_DATA[-1].exists():
    print(f'FOLDER_MQ_TXT_DATA = {folder}')
    FOLDER_MQ_TXT_DATA = folder

if not FOLDER_MQ_TXT_DATA:
    print(
        'Not found. Check FOLDER_MQ_TXT_DATA entries above: {}'.format(
            ", ".join([str(fname) for fname in FOLDERS_MQ_TXT_DATA])
        )
    )
    FOLDER_MQ_TXT_DATA = FOLDERS_MQ_TXT_DATA[1]
    FOLDER_MQ_TXT_DATA.mkdir()
    ON_ERDA = False
    print(f"Created local folder: {FOLDER_MQ_TXT_DATA}")

if ON_ERDA:
    import sys
    sys.path.append('/home/jovyan/work/vaep/')

    FOLDER_MQ_TXT_DATA = Path('/home/jovyan/work/mq_out/')
    if FOLDER_MQ_TXT_DATA.exists():
        print(f'FOLDER_MQ_TXT_DATA = {FOLDER_MQ_TXT_DATA}')
    else:
        raise FileNotFoundError(f"Check config for FOLDER_MQ_TXT_DATA")

    FOLDER_RAW_DATA = Path('/home/jovyan/work/share_hela_raw/')
    if FOLDER_RAW_DATA.exists():
        print(f'FOLDER_RAW_DATA = {FOLDER_RAW_DATA}')
    else:
        raise FileNotFoundError(
            f"Check config for FOLDER_RAW_DATA: {FOLDER_RAW_DATA}")

# FOLDER_KEY  = None

FOLDER_KEY = 'txt'

###############################################################################
###############################################################################
# Files

FN_ALL_SUMMARIES = FOLDER_PROCESSED / 'all_summaries.json'

#FN_PEPTIDE_INTENSITIES = Path(FOLDER_DATA) / 'mq_out' / 'peptide_intensities.pkl'
FN_PEPTIDE_STUMP = 'peptide_intensities'
FN_PEPTIDE_INTENSITIES = Path(FOLDER_DATA) / 'peptide_intensities.pkl'

FN_PROTEIN_TSV = FOLDER_DATA / 'Mann_Hepa_data.tsv'

PREFIX_IMPUTED = 'hela_imputed_proteins'
PREFIX_META = 'hela_metadata'

FOLDER_FASTA = Path(FOLDER_DATA) / 'fasta'
FN_FASTA_DB = FOLDER_FASTA / 'fasta_db.json'
FN_ID_MAP = FOLDER_FASTA / 'id_map.json'
FN_PROT_GENE_MAP = FOLDER_FASTA / 'uniprot_protein_gene_map.json'
FN_PEP_TO_PROT = FOLDER_FASTA / 'peptided_to_prot_id.json'
FN_PROTEIN_SUPPORT_MAP = Path(FOLDER_DATA) / 'protein_support.pkl'
FN_PROTEIN_SUPPORT_FREQ = Path(FOLDER_DATA) / 'dict_protein_support_freq.pkl'

FN_ALL_RAW_FILES = 'all_raw_files_dump.txt'

# DATA FASTA Config
KEY_FASTA_HEADER = 'meta'
KEY_FASTA_SEQ = 'seq'
KEY_PEPTIDES = 'peptides'
KEY_GENE_NAME = 'gene'
KEY_GENE_NAME_FASTA = 'gene_fasta'

KEYS_FASTA_ENTRY = [KEY_FASTA_HEADER,
                    KEY_FASTA_SEQ, KEY_PEPTIDES, KEY_GENE_NAME]

FastaEntry = namedtuple('FastaEntry', KEYS_FASTA_ENTRY)
fasta_entry = FastaEntry(*KEYS_FASTA_ENTRY)


FILEPATH_UTILS = 'src/file_utils.py'

FNAME_C_PEPTIDES = FOLDER_PROCESSED / \
    'count_all_peptides.json'  # aggregated peptides
# evidence peptides (sequence, charge, modification)
FNAME_C_EVIDENCE = FOLDER_PROCESSED / 'count_all_evidences.json'

FNAME_C_PG = FOLDER_PROCESSED / 'count_all_protein_groups.json'
FNAME_C_GENES = FOLDER_PROCESSED / 'count_all_genes.json'


def build_df_fname(df: pandas.DataFrame, stub: str) -> str:
    N, M = df.shape
    return f'{stub}_N{N:05d}_M{M:05d}'


def insert_shape(df: pandas.DataFrame, template: str = "filename{}.txt", shape=None):
    if shape is None:
        N, M = df.shape
    else:
        N, M = shape
    return template.format(f'_N{N:05d}_M{M:05d}')

# put to testing
# df_test = pd.DataFrame(np.random.randint(low=-4, high=10, size=(1729, 146)))
# N, M = df_test.shape
# assert build_fname(df=df_test, stub='df_intensities') == f'df_intensities_N{N:05d}_M{M:05d}'


###############################################################################
###############################################################################
# configure plotting
# https://matplotlib.org/stable/users/dflt_style_changes.html
mpl.rcParams['figure.figsize'] = [10.0, 8.0]

# cfg.keys.gene_name
# cfg.paths.processed
# cfg.


class Config():
    """Config class with a setter enforcing that config entries cannot 
    be overwritten.


    Can contain configs, which are itself configs:
    keys, paths,
    
    """

    def __setattr__(self, entry, value):
        """Set if attribute not in instance."""
        if hasattr(self, entry) and getattr(self, entry) != value:
            raise AttributeError(
                f'{entry} already set to {getattr(self, entry)}')
        super().__setattr__(entry, value)

    def __repr__(self):
        return pformat(vars(self))  # does not work in Jupyter?

    def overwrite_entry(self, entry, value):
        """Explicitly overwrite a given value."""
        super().__setattr__(entry, value)

    def dump(self, fname=None):
        if fname is None:
            try:
                fname = self.out_folder
                fname = Path(fname) / 'model_config.yml'
            except AttributeError:
                raise AttributeError(
                    'Specify fname or set "out_folder" attribute.')
        d = vaep.io.parse_dict(input_dict=self.__dict__)
        with open(fname, 'w') as f:
            yaml.dump(d, f)
        logger.info(f"Dumped config to: {fname}")


if __name__ == '__main__':
    cfg = Config()
    cfg.test = 'test'
    print(cfg.test)
    cfg.test = 'raise ValueError'
