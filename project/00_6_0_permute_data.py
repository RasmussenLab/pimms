# %% [markdown]
# # Permute featues in data


# %%
from pathlib import Path
from typing import Union, List

import numpy as np
import vaep
import vaep.analyzers.analyzers
from vaep.utils import create_random_df

logger = vaep.logging.setup_nb_logger()
logger.info("Split data and make diagnostic plots")

# %%
t = create_random_df(N=10, M=3)
t = t.apply(lambda x: np.arange(len(x)))
t

# %%
rng = np.random.default_rng()
t.apply(rng.permutation).sort_values(by='feat_0')

# %%
# catch passed parameters
args = None
args = dict(globals()).keys()


# %% tags=["parameters"]
FN_INTENSITIES: str = 'data/dev_datasets/df_intensities_proteinGroups_long/Q_Exactive_HF_X_Orbitrap_6070.pkl'  # Sample (rows) intensiites for features (columns)
index_col: Union[str, int] = 0  # Can be either a string or position (typical 0 for first column), or a list of these.
column_names: List[str] = ["Gene Names"]  # Manuelly set column names (of Index object in columns)
out_folder: str = ''  # Output folder for permuted data, optional. Default is to save with suffix '_permuted' in same folder as input data
random_seed: int = 42  # Random seed for reproducibility
file_format: str = 'pkl'

# %%
args = vaep.nb.get_params(args, globals=globals())
args

# %%
args = vaep.nb.Config().from_dict(args)
args


# %%
if isinstance(args.index_col, str) or isinstance(args.index_col, int):
    args.overwrite_entry('index_col', [args.index_col])
args.index_col  # make sure it is an iterable

# %% [markdown]
# ## Raw data

# %% [markdown]
# process arguments

# %%
logger.info(f"{args.FN_INTENSITIES = }")


FILE_FORMAT_TO_CONSTRUCTOR_IN = {'csv': 'from_csv',
                                 'pkl': 'from_pickle',
                                 'pickle': 'from_pickle',
                                 }

FILE_EXT = Path(args.FN_INTENSITIES).suffix[1:]
logger.info(f"File format (extension): {FILE_EXT}  (!specifies data loading function!)")

# %%
constructor = getattr(
    vaep.analyzers.analyzers.AnalyzePeptides,
    FILE_FORMAT_TO_CONSTRUCTOR_IN[FILE_EXT])  # AnalyzePeptides.from_csv
analysis = constructor(fname=args.FN_INTENSITIES,
                       index_col=args.index_col,
                       )


# %%
analysis.df.iloc[:10, :5]

# %%
rng = np.random.default_rng(seed=args.random_seed)
df = analysis.df.apply(rng.permutation)

df.iloc[:10, :5]
# %%
FILE_FORMAT_TO_CONSTRUCTOR = {'csv': 'to_csv',
                              'pkl': 'to_pickle',
                              'pickle': 'to_pickle',
                              }

method = getattr(df, FILE_FORMAT_TO_CONSTRUCTOR.get(FILE_EXT))

fname = vaep.utils.append_to_filepath(args.FN_INTENSITIES, 'permuted')
method(fname)
# %%
constructor = getattr(
    vaep.analyzers.analyzers.AnalyzePeptides,
    FILE_FORMAT_TO_CONSTRUCTOR_IN[FILE_EXT])  # AnalyzePeptides.from_csv
analysis = constructor(fname=args.FN_INTENSITIES,
                       index_col=args.index_col,
                       )
