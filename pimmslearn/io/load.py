import logging
from typing import Union, List

import pandas as pd

logger = logging.getLogger(__name__)


def verify_df(df: pd.DataFrame,
              fname: str,
              index_col: str,  # could be potentially 0 for the first column
              verify_fname: bool = False,
              usecols=None,
              ):
    if usecols and isinstance(index_col, str):
        assert index_col in usecols, 'Add index_col to usecols Sequence'
    if verify_fname:
        if not len(df.shape) == 2:
            raise ValueError(f"Expected 2 -dimensional array, not {len(df.shape)} -dimensional,"
                             f" of type: {type(df)}")
        N, M = df.shape
        assert f'N{N:05d}' in str(fname) and f'M{M:05d}' in str(fname), \
            ("Filename number don't match loaded numbers: "
                f"{fname} should contain N{N} and M{M}")


def from_csv(fname: str,
             nrows: int = None,
             # could be potentially 0 for the first column
             index_col: Union[int, str, List] = 'Sample ID',
             verify_fname: bool = False,
             usecols=None,
             **kwargs):
    logger.warning(f"Passed unknown kwargs: {kwargs}")
    df = pd.read_csv(fname, index_col=index_col, low_memory=False,
                     nrows=nrows, usecols=usecols).squeeze('columns')
    if len(df.shape) == 1:
        # unstack all but first column
        df = df.unstack(df.index.names[1:])
    verify_df(df=df, fname=fname,
              index_col=index_col,
              verify_fname=verify_fname,
              usecols=usecols)
    return df  # all __init__ parameters are kwargs


def from_pickle(fname: str,
                # could be potentially 0 for the first column
                index_col: Union[int, str, List] = 'Sample ID',
                verify_fname: bool = False,
                usecols=None,
                **kwargs) -> pd.DataFrame:
    logger.warning(f"Passed unknown kwargs: {kwargs}")
    df = pd.read_pickle(fname).squeeze()
    if len(df.shape) == 1:
        df = df.unstack(df.index.names[1:])
    verify_df(df=df, fname=fname,
              index_col=index_col,
              verify_fname=verify_fname,
              usecols=usecols)
    return df  # all __init__ parameters are kwargs
