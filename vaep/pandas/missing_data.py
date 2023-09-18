from __future__ import annotations
from pathlib import Path
import math

import pandas as pd


def percent_missing(df: pd.DataFrame) -> float:
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
