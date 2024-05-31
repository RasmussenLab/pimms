"""Download the development dataset of HeLa cells from PRIDE.

Instrument: Q_Exactive_HF_X_Orbitrap_6070

Can be adapted to save all instruments or other datasets.
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

FTP_FOLDER = 'https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/12/PXD042233'
FILE = 'pride_metadata.csv'
print(f'Fetch metadata: {FTP_FOLDER}/{FILE}')
meta = pd.read_csv(f'{FTP_FOLDER}/{FILE}', index_col=0)
meta.sample(5, random_state=42).sort_index()
idx_6070 = meta.query('`instrument serial number`.str.contains("#6070")').index

FILE = 'geneGroups_aggregated.zip'
print(f"Fetch archive:  {FTP_FOLDER}/{FILE}")
r = requests.get(f'{FTP_FOLDER}/{FILE}', timeout=900)
with zipfile.ZipFile(io.BytesIO(r.content), 'r') as zip_archive:
    print('available files in archive' '\n - '.join(zip_archive.namelist()))
    FNAME = 'geneGroups/intensities_wide_selected_N07444_M04547.csv'
    print('\nread file:', FNAME)
    with zip_archive.open(FNAME) as f:
        df = pd.read_csv(f, index_col=0)

# dev_datasets/df_intensities_proteinGroups_long/Q_Exactive_HF_X_Orbitrap_6070.pkl
FOLDER = Path('dev_datasets/df_intensities_proteinGroups_long')
FOLDER.mkdir(parents=True, exist_ok=True)
fname = FOLDER / 'Q_Exactive_HF_X_Orbitrap_6070.csv'
df.loc[idx_6070].to_csv(fname)
print(f'saved data to: {fname}')
df.loc[idx_6070].to_pickle(fname.with_suffix('.pkl'))
print(f'saved data to: {fname.with_suffix(".pkl")}')
# save metadata:
fname = FOLDER / 'metadata.csv'
meta.loc[idx_6070].to_csv(fname)
print(f'saved metadata to: {fname}')
