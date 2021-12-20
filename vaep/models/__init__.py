from functools import reduce
import logging
from operator import mul
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastcore.foundation import L
from fastai import learner

import sklearn.metrics as sklm


from . import ae

logger = logging.getLogger(__name__)


def plot_loss(recorder: learner.Recorder,
              skip_start: int = 5,
              with_valid: bool = True,
              ax: plt.Axes = None) -> plt.Axes:
    """Adapted Recorder.plot_loss to accept matplotlib.axes.Axes argument.
    Allows to build combined graphics.

    Parameters
    ----------
    recorder : learner.Recorder
        fastai Recorder object, learn.recorder
    skip_start : int, optional
        Skip N first batch metrics, by default 5
    with_valid : bool, optional
        Add validation data loss, by default True
    ax : plt.Axes, optional
        Axes to plot on, by default None

    Returns
    -------
    plt.Axes
        [description]
    """
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(list(range(skip_start, len(recorder.losses))),
            recorder.losses[skip_start:], label='train')
    if with_valid:
        idx = (np.array(recorder.iters) < skip_start).sum()
        ax.plot(recorder.iters[idx:], L(
            recorder.values[idx:]).itemgot(1), label='valid')
        ax.legend()
    return ax


def split_prediction_by_mask(pred: pd.DataFrame,
                             mask: pd.DataFrame,
                             check_keeps_all: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    pred : pd.DataFrame
        prediction DataFrame
    mask : pd.DataFrame
        Mask with same indices as pred DataFrame.
    check_keeps_all : bool, optional
        if True, perform sanity checks, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        prediction for inversed mask, and predicitions for mask
    """
    test_pred_observed = pred[~mask].stack()
    test_pred_real_na = pred[mask].stack()
    if check_keeps_all:
        assert len(test_pred_real_na) + \
            len(test_pred_observed) == reduce(mul, pred.shape)
    return test_pred_observed, test_pred_real_na


def compare_indices(first_index: pd.Index, second_index: pd.Index) -> pd.Index:
    """Show difference of indices in other index wrt. to first. First should be the larger
    collection wrt to the second. This is the set difference of two Index objects.
    
    If second index is a superset of indices of the first, the set will be empty,
    although there  are differences (default behaviour in pandas).
    
    Parameters
    ----------
    first_index : pd.Index
        Index, should be superset
    second_index : pd.Index
        Index, should be the subset
        
    Returns
    -------
    pd.Index
        Return a new Index with elements of the first index not in second.
    """
    _diff_index = first_index.difference(second_index)
    if len(_diff_index):
        print("Some predictions couldn't be generated using the approach using artifical replicates.\n"
              "These will be omitted for evaluation.")
        for _index in _diff_index:
            print(f"{_index[0]:<40}\t {_index[1]:<40}")
    return _diff_index


scoring = [('MSE', sklm.mean_squared_error),
           ('MAE', sklm.mean_absolute_error)]


def get_metrics_df(pred_df: pd.DataFrame,
                   true_col: List[str] = None,
                   scoring: List[Tuple[str, Callable]] = scoring) -> pd.DataFrame:
    """Create metrics based on predictions, a truth reference and a
    list of scoring function with a name.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame containing `true_col`.
    true_col : List[str], optional
        Column of ground truth values, by default None
    scoring : List[Tuple[str, Callable]], optional
        List of tuples. A tuple is a set of (key, funtion) pairs.
        The function take y_true and y_pred - as for all sklearn metrics, by default scoring

    Returns
    -------
    pd.DataFrame
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if not true_col:
        # assume first column is truth if None is given
        y_true = pred_df.iloc[:, 0]
        print(f'Selected as truth to compare to: {y_true.name}')
        y_pred = pred_df.iloc[:, 1:]
    else:
        if issubclass(type(true_col), int):
            y_true = pred_df.iloc[:, true_col]
            pred_df = pred_df.drop(y_true.name, axis=1)
        elif issubclass(type(true_col), str):
            y_true = pred_df[true_col]
            pred_df = pred_df.drop(true_col, axis=1)
        else:
            raise ValueError(
                f'true_col has to be of type str or int, not {type(true_col)}')
    if y_true.isna().any():
        raise ValueError(f"Ground truth column '{y_true.name}' contains missing values. "
                         "Drop these rows first.")
    # # If NAs in y_true should be allowed, the intersection between pred columns
    # # and y_true has to be added in the for loop below.
    # if y_true.isna().any():
        # logger.info(f"Remove {y_true.isna().sum()} from true_col {y_true.name}")
        # y_true = y_true.dropna()
        # assert len(y_true)
        
    metrics = {}
    for model_key in y_pred:
        model_pred = y_pred[model_key]
        model_pred_no_na = model_pred.dropna()
        if len(model_pred) > len(model_pred_no_na):
            logger.info(
                f"Drop indices for {model_key}: "
                "{}".format([(idx[0], idx[1])
                             for idx
                             in model_pred.index.difference(model_pred_no_na.index)]))

        metrics[model_key] = dict(
            [(k, f(y_true=y_true.loc[model_pred_no_na.index], y_pred=model_pred_no_na))
             for k, f in scoring]
        )
    metrics = pd.DataFrame(metrics)
    return metrics
