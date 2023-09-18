# %% [markdown]
# # Submission file


# %%
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path, PurePosixPath


# %%
# Parameters
FOLDER = Path('data/rename')

# %%
file = FOLDER / 'files_on_pride.log'
# file = FOLDER / 'files_pride_server_toplevel.log'

# %%
counts = defaultdict(int)
with open(file) as f:
    for line in f:
        fname = line.strip()
        suffix = PurePosixPath(fname).suffix
        counts[suffix] += 1
dict(counts)

# %% [markdown]
# Only create a few files for creation a submission.px template...
#
# # %%
# SUBMISSON_FOLDER = Path('data/rename/submission')
# SUBMISSON_FOLDER.mkdir(exist_ok=True)
# with open(file) as f:
#     hash = 'placeholder'
#     for line in f:
#         # fname = line.strip().split()
#         fname = line.strip()
#         fname = PurePosixPath(fname).name
#         with open(SUBMISSON_FOLDER / fname, 'w') as f_out:
#             f_out.write(f'{hash}  {fname}')
# # %%
# files = list(SUBMISSON_FOLDER.iterdir())
# print(f"{len(files) = :,d}")

# %% [markdown]
# 7444 raw files
# 7444 zip files with MaxQuant results
# 3 zip files with aggregated MaxQuant results
# 1 SDRF file as tsv
# 2 csv files with metadata of the raw files and the MaxQuant results summaries

# %%
# len(files) == 7444*2 + 6  # expected number of files

# %% [markdown]
# This was not really necessary...

# %%
file_types = {'.zip': 'SEARCH',
              '.raw': 'RAW',
              '.csv': 'SEARCH',
              '.tsv': 'EXPERIMENTAL_DESIGN'}

# %%
files = pd.DataFrame(columns='FMH	file_id	file_type	file_path	file_mapping'.split('\t'))
files['file_path'] = pd.read_csv(file, header=None)
files['FMH'] = 'FMH'
files['file_id'] = files.index
files['file_type'] = files['file_path'].map(lambda x: file_types[Path(x).suffix])
files['file_mapping'] = files['file_id'] - 1
files.loc[
    files['file_type'] != 'SEARCH', 'file_mapping'] = np.nan
files = files.astype({'file_id': int, 'file_mapping': pd.Int32Dtype()})
files
# %%
files.to_csv(FOLDER / 'submiss.px_to_add.tsv', sep='\t', index=False)
# %% [markdown]
# Some manuel adding of the last files still required...
