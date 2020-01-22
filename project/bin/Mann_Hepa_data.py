import os
import pandas as pd
import logging
from vaep.utils import load_data

logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)
sH = logging.StreamHandler()
sH.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
# sH.setFormatter(formatter)
logger.addHandler(sH)

FOLDER = 'data'
FILE = 'Mann_Hepa_data.tsv'

_file = os.path.join(FOLDER, FILE)
# df = pd.read_csv(_file, sep='\t', index_col='index')
# meta_data = df.iloc[:,:5]
# proteins = df.iloc[:,5:]

meta_data, proteins = load_data(_file)

info_na = proteins.isna().sum().describe(percentiles=[x/100 for x in range(0,100,5)] )

n_na = proteins.isna().sum().sum()
n_total = 12026971

## Sparse Data
logger.info("Percent of missing proteins: {:.5f}".format(n_na / n_total))

## 