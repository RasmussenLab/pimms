from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing import List


def load_predictions(pred_files: List, shared_columns=['observed']):

    pred_files = iter(pred_files)
    fname = next(pred_files)
    pred = pd.read_csv(fname, index_col=[0, 1])

    for fname in pred_files:
        _pred_file = pd.read_csv(fname, index_col=[0, 1])
        if shared_columns:
            assert all(pred[shared_columns] == _pred_file[shared_columns])
            pred = pred.join(_pred_file.drop(shared_columns, axis=1))
        else:
            pred = pred.join(_pred_file)
    return pred


def load_single_csv_pred_file(fname:str|Path, value_name:str='intensity') -> pd.Series:
    """Load a single pred file from a single model.
     Last column are measurments, other are index.

    Parameters
    ----------
    fname : str | Path
        Path to csv file to be loaded
    value_name : str, optional
        name for measurments to be used, by default 'intensity'

    Returns
    -------
    pd.Series
        measurments as a single column with set indices
    """
    pred = pd.read_csv(fname) # getattr for other file formats
    pred = pred.set_index(pred.columns[:-1].tolist())
    pred = pred.squeeze()
    pred.name = value_name
    return pred