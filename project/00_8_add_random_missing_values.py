# %% [markdown]
# # Add MAR to dataset
# Add missing at randomm (MAR) to dataset

# %%
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import vaep.nb

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()

# %%
# Sample (rows) intensiites for features (columns)
fn_intensities: str = 'data/dev_datasets/HeLa_6070/protein_groups_wide_N50.csv'
index_col: Optional[Union[tuple, str]] = 0
# column index name, e.g. Protein Groups, peptides, etc.
col_name: Optional[str] = None
folder_experiment: str = f'runs/example'
sample_frac: float = .8  # fraction of intensities to keep
random_state: int = 42  # random state for reproducibility
folder_data: str = ''  # specify data directory if needed
file_format: str = 'csv'  # file format of create splits, default pickle (pkl)
out_root: Optional[str] = None  # specify output folder if needed


fn_intensities = "data/ALD_study/processed/ald_plasma_proteinGroups.pkl"

# %%
fn_intensities = Path(fn_intensities)
if not out_root:
    out_root = fn_intensities.parent
args = vaep.nb.get_params(args, globals=globals())
args = vaep.nb.args_from_dict(args)
args

# %%
file_format = args.fn_intensities.suffix
# ! Add check if knonw file format
FILE_FORMAT_TO_CONSTRUCTOR = {'.csv': 'read_csv',
                              '.pkl': 'read_pickle',
                              '.pickle': 'read_pickle',
                              }

load_fct = getattr(pd, FILE_FORMAT_TO_CONSTRUCTOR[file_format])
try:
    df = load_fct(args.fn_intensities, index_col=args.index_col)
except TypeError:
    df = load_fct(args.fn_intensities)
df

# %%
if args.col_name:
    df.columns.name = args.col_name

# %%
sampled = df.stack().sample(frac=args.sample_frac,
                            random_state=args.random_state).unstack()
sampled

# %%
fname_out = args.out_folder / "{stem}_{frac:0.2f}.pkl".format(
    stem=args.fn_intensities.stem,
    frac=args.sample_frac)
fname_out.as_posix()

# %%
sampled.to_pickle(fname_out)

# %%
