"""Functionality related to analyzing missing values in a pandas DataFrame."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import pandas as pd


def percent_missing(df: pd.DataFrame):
    """Total percentage of missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data.

    Returns
    -------
    float
        Proportion of missing values in the DataFrame.
    """
    return df.isna().sum().sum() / math.prod(df.shape)


def percent_non_missing(df: pd.DataFrame) -> float:
    return df.notna().sum().sum() / math.prod(df.shape)


def list_files(folder='.') -> list[str]:
    return [f.as_posix() for f in Path(folder).iterdir()]


def get_record(data: pd.DataFrame, columns_sample=False) -> dict:
    """Get summary record of data."""
    if columns_sample:
        M, N = data.shape
    else:
        N, M = data.shape
    N_obs = data.notna().sum().sum()
    N_mis = N * M - N_obs
    missing = N_mis / (N_obs + N_mis)
    record = dict(N=int(N),
                  M=int(M),
                  N_obs=int(N_obs),
                  N_mis=int(N_mis),
                  missing=float(missing), )
    return record


def decompose_NAs(data: pd.DataFrame,
                  level: Union[int, str],
                  label: int = 'summary') -> pd.DataFrame:
    """Decompose missing values by a level into real and indirectly imputed missing values.
    Real missing value have missing for all samples in a group. Indirectly imputed missing values
    are in MS-based proteomics data that would be imputed by the mean (or median) of the observed
    values in a group if the mean (or median) is used for imputation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with samples in columns and features in rows.
    level : Union[int, str]
        Index level to group by. Examples: Protein groups, peptides or precursors in MS data.
    label : int, optional
        Column name of single column dataframe returned, by default 'summary'

    Returns
    -------
    pd.DataFrame
        One column DataFrame with summary information about missing values.
    """

    real_mvs = 0
    ii_mvs = 0

    grouped = data.groupby(level=level)
    for _, _df in grouped:
        if len(_df) == 1:
            # single precursors -> all RMVs
            real_mvs += _df.isna().sum().sum()
        elif len(_df) > 1:
            # caculate the number of missing values for samples where one precursor was observed
            total_NAs = _df.isna().sum().sum()
            M = len(_df)  # normally 2 or 3
            _real_mvs = _df.isna().all(axis=0).sum() * M
            real_mvs += _real_mvs
            ii_mvs += (total_NAs - _real_mvs)
        else:
            ValueError("Something went wrong")
    assert data.isna().sum().sum() == real_mvs + ii_mvs
    return pd.Series(
        {'total_obs': data.notna().sum().sum(),
         'total_MVs': data.isna().sum().sum(),
         'real_MVs': real_mvs,
         'indirectly_imputed_MVs': ii_mvs,
         'real_MVs_ratio': real_mvs / data.isna().sum().sum(),
         'indirectly_imputed_MVs_ratio': ii_mvs / data.isna().sum().sum(),
         'total_MVs_ratio': data.isna().sum().sum() / data.size
         }).to_frame(name=label).T.convert_dtypes()
