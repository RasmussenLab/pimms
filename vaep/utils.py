import pathlib
from typing import Union
import numpy as np
import pandas as pd

from vaep.io.datasplits import long_format


def append_to_filepath(filepath: Union[pathlib.Path, str],
                       to_append: str,
                       sep: str = '_',
                       new_suffix: str = None) -> pathlib.Path:
    """Append filepath with specified to_append using a seperator.

    Example: `data.csv` to data_processed.csv
    """
    filepath = pathlib.Path(filepath)
    suffix = filepath.suffix
    if new_suffix:
        suffix = f".{new_suffix}"
    new_fp = filepath.parent / f'{filepath.stem}{sep}{to_append}{suffix}'
    return new_fp


def create_random_missing_data(N, M,
                               mean: float = 25.0, std_dev: float = 2.0,
                               prop_missing: float = 0.15):
    data = np.random.normal(loc=mean, scale=std_dev, size=(N, M))
    prop_missing = float(prop_missing)
    if prop_missing > 0.0 and prop_missing < 1.0:
        mask = np.random.choice([False, True],
                                size=data.shape,
                                p=[prop_missing, 1 - prop_missing])
        data = np.where(mask, data, np.nan)
    return data


def create_random_missing_data_long(N: int, M: int, prop_missing=0.1):
    """Build example long"""
    data = create_random_missing_data(N=N, M=M, prop_missing=prop_missing)
    df_long = long_format(pd.DataFrame(data))
    df_long.index.names = ('Sample ID', 'peptide')
    df_long.reset_index(inplace=True)
    return df_long


def create_random_df(N: int, M: int,
                     scaling_factor: float = 30.0,
                     prop_na: float = 0.0,
                     start_idx: int = 0,
                     name_index='Sample ID',
                     name_columns='peptide'):
    X = np.random.rand(N, M)

    if prop_na > 0.0 and prop_na < 1.0:
        mask = ~(X < prop_na)
        X = np.where(mask, X, np.nan)

    X *= scaling_factor

    X = pd.DataFrame(X,
                     index=[f'sample_{i:0{len(str(N))}}'
                            for i in range(start_idx, start_idx + N)],
                     columns=(f'feat_{i:0{len(str(M))}}' for i in range(M)))
    X.index.name = name_index
    X.columns.name = name_columns
    return X
