from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def load_predictions(pred_files: List, shared_columns=['observed']):

    pred_files = iter(pred_files)
    fname = next(pred_files)
    pred = pd.read_csv(fname, index_col=[0, 1])

    for fname in pred_files:
        _pred_file = pd.read_csv(fname, index_col=[0, 1])
        idx_shared = pred.index.intersection(_pred_file.index)
        assert len(idx_shared), f'No shared index between already loaded models {pred.columns} and {fname}'
        if shared_columns:
            assert all(pred.loc[idx_shared, shared_columns] == _pred_file.loc[idx_shared, shared_columns])
            pred = pred.join(_pred_file.drop(shared_columns, axis=1))
        else:
            pred = pred.join(_pred_file)
    return pred


def load_split_prediction_by_modelkey(experiment_folder: Path,
                                      split: str,
                                      model_keys: list[str],
                                      allow_missing=False,
                                      shared_columns: list[str] = None):
    """Load predictions from a list of models.

    Parameters
    ----------
    experiment_folder : Path
        Path to experiment folder
    split : str
        which split of simulated data to load
    model_keys : List
        List of model keys to be loaded
    allow_missing : bool, optional
        Ignore missing pred files of requested model, default False
    shared_columns : List, optional
        List of columns that are shared between all models, by default None

    Returns
    -------
    pd.DataFrame
        Prediction data frame with shared columns and model predictions
    """
    pred_files = [experiment_folder / 'preds' /
                  f'pred_{split}_{key}.csv' for key in model_keys]
    to_remove = list()
    for file in pred_files:
        if not file.exists():
            if allow_missing:
                print(f'WARNING: {file} does not exist')
                to_remove.append(file)
            else:
                raise FileNotFoundError(f'{file} does not exist')
    if to_remove:
        pred_files.remove(to_remove)
    return load_predictions(pred_files, shared_columns=shared_columns)


def load_single_csv_pred_file(fname: str | Path, value_name: str = 'intensity') -> pd.Series:
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
    pred = pd.read_csv(fname)  # getattr for other file formats
    pred = pred.set_index(pred.columns[:-1].tolist())
    pred = pred.squeeze()
    pred.name = value_name
    return pred
