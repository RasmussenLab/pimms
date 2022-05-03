from pathlib import Path


def mkdir(path=Path):
    path.mkdir(exist_ok=True, parents=True)
    return path


# project folder specific
FIGUREFOLDER = mkdir(Path('Figures'))
FOLDER_DATA = mkdir(Path('data'))
FOLDER_PROCESSED = mkdir(FOLDER_DATA / 'processed')
FOLDER_TRAINING = mkdir(FOLDER_DATA / 'hela_qc_data')

# (old) Synonyms
PROCESSED_DATA = FOLDER_PROCESSED
PROTEIN_DUMPS = PROCESSED_DATA