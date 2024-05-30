# Data Folder

> Put you files here.

## Download development dataset

The large development data set can be obtained from PRIDE:

```python
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ftp_folder = 'https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/12/PXD042233'
file = 'pride_metadata.csv'

meta = pd.read_csv(f'{ftp_folder}/{file}', index_col=0)
meta.sample(5, random_state=42).sort_index()
idx_6070 = meta.query('`instrument serial number`.str.contains("#6070")').index
idx_6070

file = 'geneGroups_aggregated.zip'
r = requests.get(f'{ftp_folder}/{file}')
with zipfile.ZipFile(io.BytesIO(r.content), 'r') as zip_archive:
    fname = Path('geneGroups/intensities_wide_selected_N07444_M04547.csv')
    with zip_archive.open(fname) as f:
        df = pd.read_csv(f, index_col=0)
    fname.parent.mkdir(parents=True, exist_ok=True)

# save protein groups data for instrument 6070
df.loc[idx_6070].to_csv(fname.parent / 'geneGroups_6070.csv')
```

The smaller development data set on the protein groups level is also shipped with this
repository and can be found in the [`dev_datasets/HeLa_6070`](dev_datasets/HeLa_6070/) folder.
