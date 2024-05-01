# %% [markdown]
# # Compare runtimes for different methods

# %%
from pathlib import Path
import pandas as pd

# %% tags=["parameters"]
root_folder: str = 'runs/dev_dataset_small'

# large
root_folder: str = 'runs/mnar_mcar'
root_folder: str = 'runs/appl_ald_data_2023_11/plasma'

# %%
root_folder = Path(root_folder)

# %%
# find folders in root folder and get files with tsv extension


def find_tsv_benchmarks(root_folder: Path):
    """Find snakemake benchmark files in subfolders of root_folder (pimms workflow)

    Parameters
    ----------
    root_folder : Path
        Root folder with subfolders (one level) to find benchmark files.

    Yields
    ------
    _type_
        Generator of Path objects
    """
    for folder in root_folder.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.iterdir():
            if not file.is_file():
                continue
            if file.suffix == '.tsv':
                yield file


files = find_tsv_benchmarks(root_folder)

# %%
# works for HeLa
# files = root_folder.glob('*/**/*.tsv')
# files = (x for x in files if x.is_file())

# %%
COL = 'h:m:s'  # 's' for seconds
SPLIT_TERM = '_train_'
data = dict()
for file in files:
    experiment_key, name = file.parent.name, file.stem
    if experiment_key not in data:
        data[experiment_key] = dict()
    model_key = (name
                 .split(SPLIT_TERM)[-1]
                 .split('NAGuideR_')[-1]
                 .upper()
                 )
    time = pd.read_csv(file, sep='\t')
    data[experiment_key][model_key] = time.loc[0, COL]
data = (pd
        .DataFrame(data)
        .drop('PRED')
        )
data

# %%
# see benchmark format series from snakemake
time

# %%
fname = root_folder / 'runtimes.xlsx'
data.to_excel(fname)
fname.as_posix()

# %%
runtime_dumps = [
    'runs/mnar_mcar/runtimes.xlsx',
    'runs/appl_ald_data_2023_11/plasma/runtimes.xlsx'
]
runtime_dumps = [pd.read_excel(fname, index_col=0) for fname in runtime_dumps]
runtime_dumps = pd.concat(runtime_dumps, axis=1)
runtime_dumps

# %%
fname = 'runs/runtimes.xlsx'
runtime_dumps.to_excel(fname)
fname

# %%
