import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd

from vaep.io.datasplits import long_format


def create_random_missing_data(N, M,
                               mean: float = 25.0, std_dev: float = 2.0,
                               prop_missing: float = 0.15):
    data = np.random.normal(loc=mean, scale=std_dev, size=(N, M))
    prop_missing = float(prop_missing)
    if prop_missing > 0.0 and prop_missing < 1.0:
        mask = np.random.choice([False, True],
                                size=data.shape,
                                p=[prop_missing, 1 - prop_missing])
        # mask = np.full(N*M, True)
        # mask[:int(N*M*prop_missing)] = False
        # np.random.shuffle(mask)
        # mask = mask.reshape(N, M)
        data = np.where(mask, data, np.nan)
    return data


def create_long_df(N: int, M: int, prop_missing=0.1):
    """Build example long"""
    df_long = long_format(pd.DataFrame(data))
    df_long.index.names = ('Sample ID', 'peptide')
    df_long.reset_index(inplace=True)
    return df_long


def create_DataFrame():
    data = np.arange(100).reshape(-1, 5)
    data = pd.DataFrame(data,
                        index=(f'row_{i:02}' for i in range(data.shape[0])),
                        columns=(f'feat_{i:02}' for i in range(data.shape[1]))
                        )
    return data
